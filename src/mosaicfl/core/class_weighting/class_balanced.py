"""
class_balanced.py — Estratégia padrão do projeto: peso inversamente proporcional à
frequência LOCAL da classe (He & Garcia, 2009, "Learning from Imbalanced Data", IEEE
TKDE — ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14.6).

Comportamento idêntico ao que client.py::_compute_class_weights() sempre fez, só
extraído pra uma estratégia nomeada — classes ausentes localmente recebem peso 0.0
(não usar count=1 como fallback: infla o peso pra total/(n*1) >> peso das classes
presentes, distorcendo a loss em direção a classes inexistentes).
"""
from .base import ClassWeightStrategy


class ClassBalancedStrategy(ClassWeightStrategy):
    def weight_for(
        self, *, class_idx: int, class_name: str, count: int, total: int, num_classes: int,
    ) -> float:
        if count <= 0:
            return 0.0
        return total / (num_classes * count)
