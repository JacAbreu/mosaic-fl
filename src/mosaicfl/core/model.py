"""
Modelo BEHRT simplificado para sequências temporais de exames clínicos (SimplifiedBEHRT).

Arquitetura: embedding de tokens clínicos + PositionalEncoding sinusoidal +
N camadas BEHRTEncoderLayer (Transformer com atenção multi-cabeça) + classificador linear.

Tarefa: classificação multiclasse de prognóstico clínico em 4 classes (MODEL_CFG.num_classes=4):
  0 = alta  |  1 = internacao_prolongada  |  2 = uti  |  3 = obito

Tokens de entrada: sequência temporal de analitos laboratoriais gerada pelo SequencePipeline.
  PAD=0 (padding)  |  UNK=1  |  CLS=2  |  vocab de analitos: ids 3 em diante

BEHRTEncoderLayer substitui nn.TransformerEncoderLayer para expor os pesos de
atenção por cabeça (need_weights=True, average_attn_weights=False), permitindo
análise de interpretabilidade via BEHRTPatternExtractor sem impacto no treino normal.

Pooling: CLS token (use_cls_token=True, padrão) ou masked mean sobre tokens não-PAD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from .config import MODEL_CFG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MODEL_CFG.max_seq_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class BEHRTEncoderLayer(nn.Module):
    """
    Substitui nn.TransformerEncoderLayer para expor pesos de atenção.

    A única diferença em relação ao original é que self.self_attn é chamado
    com need_weights=True, e o resultado (attn_output, attn_weights) é
    retornado quando return_attention=True.

    Shape dos pesos retornados: (batch, num_heads, seq_len, seq_len)
    O PyTorch retorna média sobre heads por padrão (average_attn_weights=True).
    Usamos average_attn_weights=False para manter os heads separados,
    o que permite análise por cabeça de atenção no BEHRTPatternExtractor.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Normalização e dropout
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # --- Self-Attention ---
        # need_weights=True  → MHA calcula e devolve os pesos
        # average_attn_weights=False → shape (batch, heads, seq, seq) em vez de (batch, seq, seq)
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # mantém dimensão de heads
        )
        src = self.norm1(src + self.dropout1(attn_out))

        # --- Feed-Forward ---
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))

        if return_attention:
            return src, attn_weights  # attn_weights: (batch, heads, seq, seq)
        return src


class SimplifiedBEHRT(nn.Module):
    def __init__(self, use_cls_token: bool = True, demo_dim: int = 0):
        """
        Args:
            demo_dim: dimensão do vetor demográfico para late fusion.
                      0 = sem demográficos (comportamento original).
                      2 = late fusion com [age_norm, sex_binary] concatenados
                          ao CLS poolado antes do classifier head.
                      Quando demo_dim > 0 e nenhum tensor é passado em forward(),
                      as dimensões demográficas são zero-padded automaticamente.
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.demo_dim = demo_dim
        self.cls_token = nn.Parameter(torch.empty(1, 1, MODEL_CFG.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.embedding = nn.Embedding(MODEL_CFG.vocab_size, MODEL_CFG.embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(MODEL_CFG.embed_dim, MODEL_CFG.max_seq_len + 1)
        self.dropout    = nn.Dropout(MODEL_CFG.dropout)

        # Lista de camadas próprias no lugar de nn.TransformerEncoder
        self.layers = nn.ModuleList([
            BEHRTEncoderLayer(
                d_model=MODEL_CFG.embed_dim,
                nhead=MODEL_CFG.num_heads,
                dim_feedforward=MODEL_CFG.ff_dim,
                dropout=MODEL_CFG.dropout,
            )
            for _ in range(MODEL_CFG.num_layers)
        ])

        # Pré-classificador: LayerNorm + Dropout para estabilidade
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(MODEL_CFG.embed_dim),
            nn.Dropout(MODEL_CFG.dropout),
        )

        # Late fusion: classifier head recebe embed_dim + demo_dim features.
        # Quando demo_dim=0, comportamento idêntico ao original.
        classifier_input_dim = MODEL_CFG.embed_dim + demo_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(MODEL_CFG.dropout),
            nn.Linear(64, MODEL_CFG.num_classes),
        )


    def _masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calcula a média apenas sobre tokens reais (mask=False).

        Args:
            x:    (batch, seq_len, embed_dim)
            mask: (batch, seq_len) — True em posições de padding

        Returns:
            pooled: (batch, embed_dim)
        """
        # mask_expanded: (batch, seq_len, 1) — 1.0 para tokens reais, 0.0 para padding
        mask_expanded = (~mask).unsqueeze(-1).float()
        masked_sum = (x * mask_expanded).sum(dim=1)          # (batch, embed_dim)
        count = mask_expanded.sum(dim=1).clamp(min=1)        # evita divisão por zero
        return masked_sum / count

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        demographics: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:                (batch, seq_len) — índices do vocabulário
            mask:             (batch, seq_len) — True onde há padding
            demographics:     (batch, demo_dim) — features demográficas para late fusion
                              [age_norm, sex_binary] quando demo_dim=2.
                              None = sem demográficos; se demo_dim > 0, zero-pad automático.
            return_attention: se True, retorna (logits, all_attn_weights)

        Returns:
            logits            quando return_attention=False  → (batch, num_classes)
            logits, weights   quando return_attention=True
                weights shape: (num_layers, batch, num_heads, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # Máscara sobre a sequência original (antes do CLS)
        if mask is None:
            mask = (x == 0)  # True em posições de padding

        emb = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Prefixa o vetor CLS learnable (parâmetro real, recebe gradiente)
        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, -1, -1)         # (batch, 1, embed_dim)
            emb = torch.cat([cls, emb], dim=1)                       # (batch, seq_len+1, embed_dim)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)                # (batch, seq_len+1)

        emb = emb * math.sqrt(MODEL_CFG.embed_dim)
        emb = self.pos_encoder(emb)
        emb = self.dropout(emb)

        all_attn_weights = []
        out = emb
        for layer in self.layers:
            if return_attention:
                out, attn_w = layer(out, src_key_padding_mask=mask, return_attention=True)
                all_attn_weights.append(attn_w)   # (batch, heads, seq, seq)
            else:
                out = layer(out, src_key_padding_mask=mask)

        # Pooling: usa <CLS> se disponível, senão masked mean pooling
        if self.use_cls_token:
            pooled = out[:, 0]  # (batch, embed_dim) — token <CLS>
        else:
            pooled = self._masked_mean_pool(out, mask)  # (batch, embed_dim)

        pooled = self.pre_classifier(pooled)

        # Late fusion: concatena demográficos ao CLS poolado antes do classifier.
        # O Transformer aprende a representação da sequência sem interferência demográfica;
        # o classifier pondera ambas as fontes de sinal de forma independente.
        if demographics is not None:
            pooled = torch.cat([pooled, demographics.to(pooled.device)], dim=-1)
        elif self.demo_dim > 0:
            zeros = torch.zeros(
                pooled.size(0), self.demo_dim,
                device=pooled.device, dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, zeros], dim=-1)

        logits = self.classifier(pooled)

        if return_attention:
            # empilha em (num_layers, batch, heads, seq, seq)
            return logits, torch.stack(all_attn_weights, dim=0)
        return logits