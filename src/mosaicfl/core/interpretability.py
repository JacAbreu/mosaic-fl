"""
interpretability.py — Extração de padrões de atenção do modelo BEHRT treinado via FL.

Ponte entre o modelo federado convergido e o sistema RAG: identifica quais tokens
clínicos (sintomas, exames) receberam maior atenção por desfecho e converte em
perfis textuais que alimentam a base de conhecimento do RAG.

Funciona com qualquer modelo que exponha forward(x, mask, return_attention=True)
retornando (logits, attn_weights) com weights no formato
(num_layers, batch, num_heads, seq_len, seq_len).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List


class BEHRTPatternExtractor:
    """
    Extrai perfis prototípicos do BEHRT treinado via FL.
    Analisa quais tokens (sintomas/exames) recebem maior atenção por desfecho.

    Args:
        model:     instância de nn.Module com pesos já carregados.
                   Deve expor forward(x, mask, return_attention=True).
        vocab_map: dicionário {token: int} produzido por EHRPreprocessor.vocab_map
                   após chamar preprocessor.process() ou preprocessor.build_vocabulary().
    """

    def __init__(self, model: nn.Module, vocab_map: Dict[str, int]):
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
        dataloader,
        desfecho_alvo: int,
        top_n: int = 50,
    ) -> List[Dict]:
        """
        Para cada amostra do desfecho alvo, registra a atenção média por token.
        Retorna os top_n perfis mais representativos.
        """
        pattern_scores = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                mask = (batch_x == 0)
                logits, attn_weights = self.model(batch_x, mask, return_attention=True)

                # (num_layers, batch, num_heads, seq, seq) → (batch, layers, heads, seq, seq)
                attn_weights = attn_weights.permute(1, 0, 2, 3, 4)

                mask_desfecho = (batch_y == desfecho_alvo)
                if mask_desfecho.sum() == 0:
                    continue

                attn_desfecho = attn_weights[mask_desfecho]  # (N, layers, heads, seq, seq)
                mean_attn = attn_desfecho.mean(dim=(1, 2))   # (N, seq, seq)

                for i in range(mean_attn.size(0)):
                    token_importance = mean_attn[i].sum(dim=0)
                    top_indices = torch.topk(token_importance, k=5).indices
                    tokens = [self.vocab_inverse.get(idx.item(), "<UNK>") for idx in top_indices]
                    pattern_scores.append({
                        "texto": f"Paciente com {', '.join(tokens)}",
                        "desfecho": "pneumonia" if desfecho_alvo == 1 else "alta",
                        "score_atencao": token_importance.sum().item(),
                        "tokens": tokens,
                    })

        pattern_scores.sort(key=lambda x: x["score_atencao"], reverse=True)
        return pattern_scores[:top_n]

    def generate_all_profiles(
        self,
        dataloader,
        desfechos: List[int] = None,
    ) -> List[Dict]:
        """Gera perfis para todos os desfechos."""
        if desfechos is None:
            desfechos = [0, 1]
        all_patterns = []
        for d in desfechos:
            all_patterns.extend(self.extract_top_patterns(dataloader, d, top_n=50))
        return all_patterns
