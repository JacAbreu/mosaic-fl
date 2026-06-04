import torch
import numpy as np
from typing import List, Dict
from .model import SimplifiedBEHRT
from .config import *


class BEHRTPatternExtractor:
    """
    Extrai perfis prototípicos do BEHRT treinado via FL.
    Analisa quais tokens (sintomas/exames) recebem maior atenção por desfecho.

    Args:
        model:     instância de SimplifiedBEHRT com pesos já carregados.
        vocab_map: dicionário {token: int} produzido por EHRPreprocessor.vocab_map
                   após chamar preprocessor.process() ou preprocessor.build_vocabulary().
    """
    def __init__(self, model: SimplifiedBEHRT, vocab_map: Dict[str, int]):
        if not vocab_map:
            raise ValueError(
                "vocab_map está vazio. Certifique-se de chamar "
                "EHRPreprocessor.process() antes de instanciar BEHRTPatternExtractor."
            )
        self.model = model
        self.model.eval()
        self.vocab_inverse: Dict[int, str] = {v: k for k, v in vocab_map.items()}
    
    def extract_top_patterns(
        self, 
        dataloader,           # dados de treinamento (ou subset representativo)
        desfecho_alvo: int,   # 0 = alta, 1 = pneumonia, etc.
        top_n: int = 50
    ) -> List[Dict]:
        """
        Para cada amostra do desfecho alvo, registra a atenção média por token.
        Retorna os top_n perfis mais representativos.
        """
        pattern_scores = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                mask = (batch_x == 0)  # True em posições de padding
                # logits: (batch, num_classes)
                # attn_weights: (num_layers, batch, num_heads, seq_len, seq_len)
                logits, attn_weights = self.model(batch_x, mask, return_attention=True)

                # Reorganiza para (batch, num_layers, num_heads, seq, seq)
                # torch.stack já retorna (layers, batch, heads, seq, seq) → permute
                attn_weights = attn_weights.permute(1, 0, 2, 3, 4)

                # Filtra apenas amostras do desfecho desejado
                mask_desfecho = (batch_y == desfecho_alvo)
                if mask_desfecho.sum() == 0:
                    continue

                attn_desfecho = attn_weights[mask_desfecho]  # (N, layers, heads, seq, seq)

                # Média sobre camadas e cabeças → (N, seq, seq)
                mean_attn = attn_desfecho.mean(dim=(1, 2))
                
                # Para cada amostra, pega os tokens mais "olhados" (maior atenção recebida)
                for i in range(mean_attn.size(0)):
                    token_importance = mean_attn[i].sum(dim=0)  # soma sobre posições de query
                    top_indices = torch.topk(token_importance, k=5).indices
                    
                    tokens = [self.vocab_inverse.get(idx.item(), "<UNK>") for idx in top_indices]
                    texto = f"Paciente com {', '.join(tokens)}"
                    
                    pattern_scores.append({
                        "texto": texto,
                        "desfecho": "pneumonia" if desfecho_alvo == 1 else "alta",
                        "score_atencao": token_importance.sum().item(),
                        "tokens": tokens
                    })
        
        # Ordena por score de atenção e pega os top N
        pattern_scores.sort(key=lambda x: x["score_atencao"], reverse=True)
        return pattern_scores[:top_n]

    def generate_all_profiles(self, dataloader, desfechos: List[int] = [0, 1]) -> List[Dict]:
        """Gera perfis para todos os desfechos."""
        all_patterns = []
        for d in desfechos:
            patterns = self.extract_top_patterns(dataloader, d, top_n=50)
            all_patterns.extend(patterns)
        return all_patterns


# ---------------------------------------------------------------------------
# COMO USAR EM run_experiments.py (dentro do fluxo pós-FL):
# ---------------------------------------------------------------------------
#
# 1. O preprocessor já foi executado e seu vocab_map está disponível:
#
#       preprocessor = EHRPreprocessor()
#       df_proc, summary = preprocessor.process(df_raw, text_cols=[...])
#       vocab = preprocessor.vocab_map          # <-- Dict[str, int]
#
# 2. Após a convergência do federado, carregue o modelo global:
#
#       global_model = SimplifiedBEHRT()
#       # ... carrega pesos agregados do servidor ...
#
# 3. Passe o vocab_map explicitamente:
#
#       extractor = BEHRTPatternExtractor(global_model, vocab_map=vocab)
#       patterns  = extractor.generate_all_profiles(train_loader_global)
#
# 4. Agora alimente o RAG:
#
#       rag = ClinicalRAG()
#       rag.build_knowledge_base(patterns)
# ---------------------------------------------------------------------------
