"""
cost_sensitive.py — Peso explícito por classe, definido por julgamento clínico (não pela
frequência dos dados). Ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md,
seção 14: raridade estatística não é o mesmo que importância clínica (ex.: um caso raro
mas grave não deveria ser tratado como "pouco importante" só porque é infrequente).

Fundamentação: Elkan (2001), "The Foundations of Cost-Sensitive Learning", IJCAI —
seção 14.1. Ressalva registrada na mesma seção (14.3, Zheng et al. 2024): o peso só
produz ganho real se vier de julgamento clínico genuíno — se for derivado da própria
frequência dos dados, essa estratégia se torna equivalente a ClassBalancedStrategy com
um nome diferente, sem ganho nenhum.

Esta estratégia não decide o número — só aplica o valor já resolvido pelo router a
partir da configuração (ver router.py).
"""
from .base import ClassWeightStrategy


class CostSensitiveStrategy(ClassWeightStrategy):
    def __init__(self, weight: float):
        self._weight = weight

    def weight_for(
        self, *, class_idx: int, class_name: str, count: int, total: int, num_classes: int,
    ) -> float:
        return self._weight
