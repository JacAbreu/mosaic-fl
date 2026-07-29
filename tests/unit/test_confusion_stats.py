"""
Testes para mosaicfl.core.confusion_stats — matriz de confusão agregada →
precisão/recall/especificidade por classe + accuracy (p̂) com IC 95% de
Wilson. Achado 2026-07-29: só existia no Caminho A (test_loader centralizado).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.confusion_stats import derive_stats_from_confusion_matrix, wilson_score_interval


class TestWilsonScoreInterval:
    def test_perfect_accuracy_large_n_interval_near_1(self):
        lo, hi = wilson_score_interval(1.0, 1000)
        assert 0.99 <= lo <= 1.0
        assert hi == 1.0

    def test_zero_accuracy_interval_near_0(self):
        lo, hi = wilson_score_interval(0.0, 1000)
        assert lo == 0.0
        assert 0.0 <= hi <= 0.01

    def test_50_50_with_moderate_n_interval_is_symmetric_around_half(self):
        lo, hi = wilson_score_interval(0.5, 100)
        assert abs((lo + hi) / 2 - 0.5) < 0.01
        assert lo < 0.5 < hi

    def test_n_zero_returns_zero_interval(self):
        assert wilson_score_interval(0.7, 0) == (0.0, 0.0)

    def test_wider_interval_for_smaller_n(self):
        lo_small, hi_small = wilson_score_interval(0.8, 10)
        lo_large, hi_large = wilson_score_interval(0.8, 10000)
        assert (hi_small - lo_small) > (hi_large - lo_large)


class TestDeriveStatsFromConfusionMatrix:
    def test_perfect_classifier_diagonal_matrix(self):
        cm = [[10, 0, 0], [0, 20, 0], [0, 0, 5]]
        labels = ["a", "b", "c"]
        stats = derive_stats_from_confusion_matrix(cm, labels)
        assert stats["accuracy_p_hat"] == 1.0
        assert stats["n_total"] == 35
        for lbl in labels:
            assert stats["per_class_stats"][lbl]["precision"] == 1.0
            assert stats["per_class_stats"][lbl]["recall_sensitivity"] == 1.0
            assert stats["per_class_stats"][lbl]["specificity"] == 1.0

    def test_known_binary_confusion_matrix(self):
        # TP=8, FN=2, FP=3, TN=7 pra classe 0 (linha 0: [8,2], coluna 0: TP+FP=8+3=11)
        cm = [[8, 2], [3, 7]]
        labels = ["pos", "neg"]
        stats = derive_stats_from_confusion_matrix(cm, labels)
        # classe "pos" (índice 0): TP=8, FN=2, FP=3, TN=7
        pos = stats["per_class_stats"]["pos"]
        assert pos["precision"] == round(8 / 11, 4)
        assert pos["recall_sensitivity"] == round(8 / 10, 4)
        assert pos["specificity"] == round(7 / 10, 4)
        assert pos["support"] == 10
        assert stats["accuracy_p_hat"] == round(15 / 20, 4)

    def test_class_with_zero_support_gives_none_not_zero(self):
        """Uma classe sem nenhuma amostra real nem prevista não deve fingir
        precisão/recall = 0.0 (isso mentiria que o modelo "acertou 0%") —
        deve ser None, sinalizando "sem dado suficiente"."""
        cm = [[10, 0, 0], [0, 5, 0], [0, 0, 0]]
        labels = ["a", "b", "c"]
        stats = derive_stats_from_confusion_matrix(cm, labels)
        assert stats["per_class_stats"]["c"]["recall_sensitivity"] is None
        assert stats["per_class_stats"]["c"]["support"] == 0

    def test_includes_confusion_matrix_and_class_labels_in_output(self):
        cm = [[1, 0], [0, 1]]
        stats = derive_stats_from_confusion_matrix(cm, ["x", "y"])
        assert stats["confusion_matrix"] == cm
        assert stats["class_labels"] == ["x", "y"]

    def test_empty_matrix_does_not_crash(self):
        cm = [[0, 0], [0, 0]]
        stats = derive_stats_from_confusion_matrix(cm, ["x", "y"])
        assert stats["accuracy_p_hat"] == 0.0
        assert stats["n_total"] == 0
