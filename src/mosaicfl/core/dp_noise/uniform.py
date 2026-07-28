"""
uniform.py — Estratégia padrão do projeto: mesmo ruído gaussiano em todo o modelo,
sem distinção de camada. Extração exata do que apply_dp_noise() sempre fez
(experiments/training/core/fl_core/aggregation.py, McMahan et al. 2018).
"""
from .base import DPNoiseStrategy


class UniformNoiseStrategy(DPNoiseStrategy):
    def group_for_key(self, key: str) -> str:
        return "all"

    def multiplier_for_group(self, group: str, base_noise_multiplier: float) -> float:
        return base_noise_multiplier
