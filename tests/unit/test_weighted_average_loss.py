import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.server_v2 import weighted_average_loss


class TestWeightedAverageLoss:
    """
    Antes desta correção, weighted_average tentava acessar m["accuracy"] em métricas
    de fit (que retornam {"loss": float}) e retornava {} silenciosamente.
    """

    def test_correct_computation(self):
        metrics = [(100, {"loss": 0.4}), (200, {"loss": 0.2})]
        result = weighted_average_loss(metrics)
        expected = (100 * 0.4 + 200 * 0.2) / 300
        assert abs(result["loss"] - expected) < 1e-6

    def test_empty_returns_empty_dict(self):
        assert weighted_average_loss([]) == {}

    def test_zero_examples_returns_zero(self):
        result = weighted_average_loss([(0, {"loss": 0.5})])
        assert result == {"loss": 0.0}

    def test_single_client(self):
        result = weighted_average_loss([(50, {"loss": 0.35})])
        assert abs(result["loss"] - 0.35) < 1e-6

    def test_returns_loss_key_not_accuracy(self):
        result = weighted_average_loss([(100, {"loss": 0.3})])
        assert "loss" in result
        assert "accuracy" not in result

    def test_fit_metrics_dict_does_not_raise(self):
        fit_metrics = [(100, {"loss": 0.42}), (150, {"loss": 0.38})]
        result = weighted_average_loss(fit_metrics)
        assert result != {}
        assert "loss" in result

    def test_missing_key_defaults_to_zero(self):
        metrics = [(100, {"loss": 0.4}), (100, {})]
        result = weighted_average_loss(metrics)
        expected = (100 * 0.4 + 100 * 0.0) / 200
        assert abs(result["loss"] - expected) < 1e-6
