"""
apply.py — Adiciona ruído gaussiano DP-FedAvg (McMahan et al. 2018) ao state_dict
agregado, roteado por estratégia (uniforme ou por grupo de camada). Mecanismo
compartilhado entre Caminho A (experiments/training/core/fl_core/aggregation.py) e
Caminho B (infrastructure/mosaicfl_server/strategy/core.py) — "o que tem no Caminho A
tem que ter no Caminho B" (decisão da autora, 2026-07-28).
"""
import logging
import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

from .base import DPNoiseStrategy
from .uniform import UniformNoiseStrategy

logger = logging.getLogger(__name__)


def apply_dp_noise(
    global_state: "OrderedDict[str, torch.Tensor]",
    round_num: int,
    n_clients: int,
    noise_multiplier: float,
    max_grad_norm: float,
    strategy: Optional[DPNoiseStrategy] = None,
    delta: float = 1e-5,
) -> Tuple[float, Dict[str, float]]:
    """Muta `global_state` in-place (mesmo comportamento histórico) e retorna
    (epsilon_simples_acumulado, multiplicadores_por_grupo_usados).

    epsilon_simples usa o PIOR CASO entre os grupos com ruído > 0 (multiplicador mais
    baixo = proteção mais fraca) — cota superior conservadora, não uma composição
    exata entre grupos heterogêneos. Grupos com multiplicador 0.0 (ex.: buffer
    excluído) não entram nessa conta — não são cobertos por DP, por desenho (não são
    sensíveis, não carregam informação aprendida do dado).
    """
    strategy = strategy or UniformNoiseStrategy()
    group_multipliers = strategy.group_multipliers(global_state.keys(), noise_multiplier)

    with torch.no_grad():
        for key in global_state:
            group = strategy.group_for_key(key)
            effective_multiplier = strategy.multiplier_for_group(group, noise_multiplier)
            noise_std = effective_multiplier * max_grad_norm / max(n_clients, 1)
            noise = torch.normal(
                0.0, noise_std, size=global_state[key].shape, device=global_state[key].device
            )
            global_state[key] = (global_state[key].float() + noise).to(global_state[key].dtype)

    protected = [m for m in group_multipliers.values() if m > 0]
    weakest_multiplier = min(protected) if protected else noise_multiplier
    eps_per_round = math.sqrt(2 * math.log(1.25 / delta)) / weakest_multiplier
    eps_accumulated = eps_per_round * round_num

    logger.info(
        "dp_noise strategy=%s groups=%s S=%.2f n=%d | "
        "ε_rodada≈%.3f ε_acum≈%.3f δ=%.0e (cota superior — pior caso entre grupos)",
        type(strategy).__name__, {g: round(m, 4) for g, m in group_multipliers.items()},
        max_grad_norm, n_clients, eps_per_round, eps_accumulated, delta,
    )
    return eps_accumulated, group_multipliers
