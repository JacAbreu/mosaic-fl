"""
router.py — Roteia cada classe pra sua estratégia de peso (Strategy pattern, decisão
2026-07-27 — ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14
"Decisão"). Uma classe com override explícito em `overrides` usa CostSensitiveStrategy;
todas as outras caem em ClassBalancedStrategy — o comportamento padrão do projeto desde
sempre. `overrides` vazio (default) preserva o comportamento atual exatamente, classe por
classe — nenhuma implementação anterior é substituída, só uma rota nova fica disponível.
"""
import logging
from collections import Counter
from typing import Dict, Optional, Sequence

import torch

from .class_balanced import ClassBalancedStrategy
from .cost_sensitive import CostSensitiveStrategy

logger = logging.getLogger(__name__)

# Teto de estabilidade — aplicado a QUALQUER estratégia, não só class_balanced: peso 47
# no BPSP (achado histórico, class_balanced sem teto) já causou explosão de gradiente.
# Um valor de custo explícito alto o bastante tem o mesmo risco — não confiar cegamente
# no julgamento clínico sem validar estabilidade, mesmo sendo intencional.
DEFAULT_CLASS_WEIGHT_CLAMP = 15.0

_BALANCED = ClassBalancedStrategy()


def compute_class_weights(
    counts: Counter,
    class_labels: Sequence[str],
    overrides: Optional[Dict[str, float]] = None,
    clamp_max: float = DEFAULT_CLASS_WEIGHT_CLAMP,
) -> torch.Tensor:
    overrides = overrides or {}
    num_classes = len(class_labels)
    total = sum(counts.values()) or 1

    raw_weights = []
    modes = {}
    for idx, name in enumerate(class_labels):
        count = counts.get(idx, 0)
        if name in overrides:
            strategy = CostSensitiveStrategy(overrides[name])
            modes[name] = "cost_sensitive"
        else:
            strategy = _BALANCED
            modes[name] = "class_balanced"
        raw_weights.append(
            strategy.weight_for(class_idx=idx, class_name=name, count=count, total=total, num_classes=num_classes)
        )

    weights = torch.tensor(raw_weights, dtype=torch.float).clamp(max=clamp_max)
    logger.info(
        "class_weight_routing modes=%s counts=%s weights=%s",
        modes, dict(counts), [round(w, 3) for w in weights.tolist()],
    )
    return weights
