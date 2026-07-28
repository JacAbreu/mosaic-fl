"""
base.py — Interface comum das estratégias de ruído DP-FedAvg (McMahan et al. 2018).

Strategy pattern (decisão 2026-07-28, mesmo princípio de mosaicfl.core.class_weighting:
nunca substituir uma abordagem já validada, sempre coexistir, roteado por config).
Uma estratégia decide, pra cada chave do state_dict agregado, qual multiplicador de
ruído gaussiano efetivo usar — UniformNoiseStrategy (padrão, todo o modelo recebe o
mesmo ruído, comportamento histórico do projeto desde sempre) ou LayerGroupNoiseStrategy
(opcional, ruído diferenciado por grupo de camada).
"""
from abc import ABC, abstractmethod
from typing import Dict, Iterable


class DPNoiseStrategy(ABC):
    @abstractmethod
    def group_for_key(self, key: str) -> str:
        """Classifica uma chave do state_dict agregado num grupo nomeado."""

    @abstractmethod
    def multiplier_for_group(self, group: str, base_noise_multiplier: float) -> float:
        """Multiplicador de ruído (σ) efetivo pra esse grupo. 0.0 é válido — significa
        'não adiciona ruído nenhum' (ex.: buffer não-treinável, não é sensível)."""

    def group_multipliers(self, keys: Iterable[str], base_noise_multiplier: float) -> Dict[str, float]:
        """Nome do grupo -> multiplicador efetivo, pra cada grupo distinto presente em
        `keys`. Usado pra contabilidade RDP (uma chamada de accountant.step() por grupo
        com ruído > 0) e auditoria/log — não decide o ruído em si, isso é
        multiplier_for_group()."""
        groups = {self.group_for_key(k) for k in keys}
        return {g: self.multiplier_for_group(g, base_noise_multiplier) for g in groups}
