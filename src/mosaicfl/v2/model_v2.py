"""
Modelo BEHRT simplificado para sequências clínicas — VERSÃO CORRIGIDA.

Mudanças principais:
  1. Masked Mean Pooling: média apenas sobre tokens reais, excluindo <PAD>.
  2. Token <CLS> opcional: índice especial para classificação de sequência.
  3. Embedding scaling ajustável via config.
  4. LayerNorm pré-classificador para estabilidade.

Correção (item 3): TransformerEncoderLayer interno usa nn.MultiheadAttention mas
descarta os pesos de atenção por padrão. A solução é substituir nn.TransformerEncoderLayer
por uma camada própria (BEHRTEncoderLayer) que chama MHA com need_weights=True e
expõe os pesos. O forward aceita return_attention=True para não quebrar o uso normal
(treinamento via Flower), retornando os pesos apenas quando solicitado.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from .config import DROPOUT, EMBED_DIM, FF_DIM, MAX_SEQ_LEN, NUM_CLASSES, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
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
    def __init__(self, use_cls_token: bool = True):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM)) 
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        # +1 no vocab para o token <CLS> se habilitado
        #vocab_size = VOCAB_SIZE + 1 if use_cls_token else VOCAB_SIZE
        #self.embedding  = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        #self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_SEQ_LEN)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_SEQ_LEN + 1)
        self.dropout    = nn.Dropout(DROPOUT)

        # Lista de camadas próprias no lugar de nn.TransformerEncoder
        self.layers = nn.ModuleList([
            BEHRTEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
            )
            for _ in range(NUM_LAYERS)
        ])

        # Pré-classificador: LayerNorm + Dropout para estabilidade
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

        # Registra índice do token <CLS> no final do vocab
        if use_cls_token:
            self.register_buffer('cls_token_id', torch.tensor(VOCAB_SIZE - 1))

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
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:                (batch, seq_len) — índices do vocabulário
            mask:             (batch, seq_len) — True onde há padding
            return_attention: se True, retorna (logits, all_attn_weights)

        Returns:
            logits            quando return_attention=False  → (batch, num_classes)
            logits, weights   quando return_attention=True
                weights shape: (num_layers, batch, num_heads, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # Adiciona token <CLS> no início de cada sequência, se habilitado
        if self.use_cls_token:
            cls_tokens = self.cls_token_id.expand(batch_size, 1)  # (batch, 1)
            x = torch.cat([cls_tokens, x], dim=1)                # (batch, seq_len+1)
            if mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)        # (batch, seq_len+1)

        # x: (batch, seq_len)
        emb = self.embedding(x)
        # Scaling opcional (configurável via SCALE_EMBEDDINGS se desejado)
        emb = emb * math.sqrt(EMBED_DIM)
        emb = self.pos_encoder(emb)
        emb = self.dropout(emb)

        if mask is None:
            mask = (x == 0)  # True em posições de padding

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
        logits = self.classifier(pooled)

        if return_attention:
            # empilha em (num_layers, batch, heads, seq, seq)
            return logits, torch.stack(all_attn_weights, dim=0)
        return logits