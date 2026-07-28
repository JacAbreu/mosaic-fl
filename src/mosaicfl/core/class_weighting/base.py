"""
base.py — Interface comum das estratégias de peso de classe.

Strategy pattern (decisão 2026-07-27, ver docs/pesquisa_baseline_implementacao_fontes_
bibliograficas.md, seção 14): cada abordagem de peso de classe vive numa implementação
própria, roteada por configuração (router.py) — nunca substitui a anterior. Uma classe
sem override cai automaticamente em ClassBalancedStrategy, preservando exatamente o
comportamento já validado do projeto.
"""
from abc import ABC, abstractmethod


class ClassWeightStrategy(ABC):
    """Calcula o peso de UMA classe pra CrossEntropyLoss(weight=...).

    Não aplica teto de estabilidade (clamp) — isso é responsabilidade do router,
    que precisa aplicar o mesmo teto a todas as classes, não importa a estratégia."""

    @abstractmethod
    def weight_for(
        self, *, class_idx: int, class_name: str, count: int, total: int, num_classes: int,
    ) -> float:
        """
        class_idx:   índice da classe (0..num_classes-1)
        class_name:  nome da classe (MODEL_CFG.class_labels[class_idx])
        count:       nº de exemplos dessa classe no loader local deste cliente
        total:       nº total de exemplos no loader local (todas as classes)
        num_classes: nº total de classes
        """
