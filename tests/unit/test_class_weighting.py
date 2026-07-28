"""
Testes para mosaicfl.core.class_weighting — Strategy pattern de peso de classe (decisão
2026-07-27, ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14):
ClassBalancedStrategy (peso por frequência local, comportamento padrão do projeto) e
CostSensitiveStrategy (peso explícito por classe, julgamento clínico), roteadas por
`overrides` — classe sem override cai em ClassBalancedStrategy, preservando exatamente
o comportamento que _compute_class_weights() sempre teve antes desta refatoração.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.class_weighting import (
    ClassBalancedStrategy,
    CostSensitiveStrategy,
    compute_class_weights,
)
from mosaicfl.core.class_weighting.router import DEFAULT_CLASS_WEIGHT_CLAMP


class TestClassBalancedStrategy:
    def test_inverse_frequency(self):
        strategy = ClassBalancedStrategy()
        # total=100, num_classes=5, count=10 -> 100/(5*10) = 2.0
        w = strategy.weight_for(class_idx=0, class_name="a", count=10, total=100, num_classes=5)
        assert w == pytest.approx(2.0)

    def test_absent_class_gets_zero(self):
        strategy = ClassBalancedStrategy()
        w = strategy.weight_for(class_idx=0, class_name="a", count=0, total=100, num_classes=5)
        assert w == 0.0

    def test_rare_class_gets_higher_weight(self):
        strategy = ClassBalancedStrategy()
        w_rare = strategy.weight_for(class_idx=0, class_name="a", count=1, total=100, num_classes=5)
        w_common = strategy.weight_for(class_idx=1, class_name="b", count=50, total=100, num_classes=5)
        assert w_rare > w_common


class TestCostSensitiveStrategy:
    def test_returns_fixed_weight_regardless_of_count(self):
        strategy = CostSensitiveStrategy(weight=25.0)
        w_rare = strategy.weight_for(class_idx=0, class_name="a", count=1, total=100, num_classes=5)
        w_common = strategy.weight_for(class_idx=1, class_name="b", count=50, total=100, num_classes=5)
        assert w_rare == 25.0
        assert w_common == 25.0

    def test_ignores_absent_class_count_zero(self):
        """Diferente de ClassBalancedStrategy: custo explícito não zera na ausência —
        é um julgamento clínico, não uma medida estatística."""
        strategy = CostSensitiveStrategy(weight=25.0)
        w = strategy.weight_for(class_idx=0, class_name="a", count=0, total=100, num_classes=5)
        assert w == 25.0


class TestComputeClassWeightsRouter:
    def test_no_overrides_matches_legacy_class_balanced_formula(self):
        """Regressão: sem overrides, o resultado deve ser idêntico à fórmula antiga de
        _compute_class_weights() (total/(n*count), 0.0 se ausente, clamp 15.0)."""
        counts = Counter({0: 50, 1: 0, 2: 10, 3: 30, 4: 10})
        class_labels = ("a", "b", "c", "d", "e")
        weights = compute_class_weights(counts, class_labels)
        total = 100
        n = 5
        expected = [
            total / (n * 50), 0.0, total / (n * 10), total / (n * 30), total / (n * 10),
        ]
        for got, exp in zip(weights.tolist(), expected):
            assert got == pytest.approx(min(exp, DEFAULT_CLASS_WEIGHT_CLAMP))

    def test_override_routes_single_class_to_cost_sensitive(self):
        counts = Counter({0: 50, 1: 1, 2: 49})
        class_labels = ("curado_pronto", "curado_internado", "melhora_pronto")
        weights = compute_class_weights(
            counts, class_labels, overrides={"curado_internado": 25.0}, clamp_max=30.0,
        )
        # classe overridden usa o valor explícito, não a frequência
        assert weights[1].item() == pytest.approx(25.0)
        # classes sem override continuam em class_balanced
        assert weights[0].item() == pytest.approx(100 / (3 * 50))
        assert weights[2].item() == pytest.approx(100 / (3 * 49))

    def test_override_still_respects_clamp(self):
        counts = Counter({0: 10})
        class_labels = ("a",)
        weights = compute_class_weights(
            counts, class_labels, overrides={"a": 999.0}, clamp_max=15.0,
        )
        assert weights[0].item() == pytest.approx(15.0)

    def test_empty_overrides_dict_same_as_none(self):
        counts = Counter({0: 10, 1: 10})
        class_labels = ("a", "b")
        w_none = compute_class_weights(counts, class_labels, overrides=None)
        w_empty = compute_class_weights(counts, class_labels, overrides={})
        assert w_none.tolist() == w_empty.tolist()

    def test_unknown_class_name_in_overrides_is_ignored(self):
        """Override pra uma classe que não existe em class_labels não deve quebrar nem
        afetar nada — é um erro de configuração silencioso, não um erro fatal."""
        counts = Counter({0: 10})
        class_labels = ("a",)
        weights = compute_class_weights(
            counts, class_labels, overrides={"classe_inexistente": 25.0},
        )
        assert weights[0].item() == pytest.approx(min(10 / (1 * 10), DEFAULT_CLASS_WEIGHT_CLAMP))
