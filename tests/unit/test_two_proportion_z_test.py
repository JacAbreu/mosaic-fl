"""
Testes para pooled_proportion() / two_proportion_z_test() — p' (proporção
agrupada) e teste-z de duas proporções, confirmado com a autora (curso de
estatística) em 2026-07-30. Compara duas proporções quaisquer com seus n's —
serve tanto pra accuracy geral (dois treinos) quanto pra recall/precisão de
UMA classe específica entre dois treinos (ex.: recall de curado_internado
antes/depois de peso de classe — a hipótese de classes raras da autora).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.confusion_stats import pooled_proportion, two_proportion_z_test


class TestPooledProportion:
    def test_equal_samples_same_proportion(self):
        assert pooled_proportion(0.8, 100, 0.8, 100) == pooled_proportion(0.8, 200, 0.8, 200)

    def test_weighted_by_sample_size(self):
        # 80% de 100 + 40% de 300 = (80+120)/400 = 0.5
        p_pooled = pooled_proportion(0.8, 100, 0.4, 300)
        assert abs(p_pooled - 0.5) < 1e-9

    def test_zero_total_n_returns_zero(self):
        assert pooled_proportion(0.5, 0, 0.5, 0) == 0.0


class TestTwoProportionZTest:
    def test_identical_proportions_gives_p_value_near_1(self):
        result = two_proportion_z_test(0.75, 500, 0.75, 500)
        assert result["z_statistic"] == 0.0
        assert result["p_value"] > 0.99
        assert result["significant"] is False

    def test_large_clear_difference_is_significant(self):
        # 90% de 1000 vs. 50% de 1000 — diferença enorme, deve ser significativa
        result = two_proportion_z_test(0.90, 1000, 0.50, 1000)
        assert result["p_value"] < 0.001
        assert result["significant"] is True
        assert result["z_statistic"] > 0

    def test_small_difference_small_n_not_significant(self):
        # diferença pequena (2pp) com amostra pequena — não deve dar significativo
        result = two_proportion_z_test(0.52, 30, 0.50, 30)
        assert result["significant"] is False

    def test_direction_of_diff_matches_sign_of_z(self):
        higher_first = two_proportion_z_test(0.8, 200, 0.6, 200)
        lower_first = two_proportion_z_test(0.6, 200, 0.8, 200)
        assert higher_first["z_statistic"] > 0
        assert lower_first["z_statistic"] < 0
        assert higher_first["diff"] == round(0.8 - 0.6, 4)

    def test_zero_n_returns_none_stats_not_crash(self):
        result = two_proportion_z_test(0.7, 0, 0.6, 100)
        assert result["p_pooled"] is None
        assert result["z_statistic"] is None
        assert result["p_value"] is None
        assert result["significant"] is None
        # p̂/n brutos continuam disponíveis mesmo sem poder testar
        assert result["p_hat_1"] == 0.7
        assert result["n1"] == 0

    def test_includes_pooled_proportion_in_result(self):
        result = two_proportion_z_test(0.8, 100, 0.4, 300)
        assert result["p_pooled"] == pytest.approx(0.5, abs=1e-4)

    def test_custom_alpha_threshold(self):
        # diferença moderada — significativa em alpha=0.10 mas não em alpha=0.001
        result_loose = two_proportion_z_test(0.55, 200, 0.45, 200, alpha=0.10)
        result_strict = two_proportion_z_test(0.55, 200, 0.45, 200, alpha=0.001)
        assert result_loose["p_value"] == result_strict["p_value"]
        assert result_loose["significant"] != result_strict["significant"] or result_loose["p_value"] < 0.001
