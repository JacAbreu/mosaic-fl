import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import weighted_average, weighted_average_accuracy


class TestWeightedAverage:

    def test_correct_computation(self):
        metrics = [(100, {"accuracy": 0.8}), (200, {"accuracy": 0.9})]
        result = weighted_average(metrics)
        expected = (100 * 0.8 + 200 * 0.9) / 300
        assert abs(result["accuracy"] - expected) < 1e-6

    def test_empty_returns_empty_dict(self):
        assert weighted_average([]) == {}

    def test_zero_examples_returns_zero(self):
        result = weighted_average([(0, {"accuracy": 0.9})])
        assert result == {"accuracy": 0.0}

    def test_single_client(self):
        result = weighted_average([(50, {"accuracy": 0.75})])
        assert abs(result["accuracy"] - 0.75) < 1e-6

    def test_equal_weights(self):
        metrics = [(100, {"accuracy": 0.6}), (100, {"accuracy": 0.8})]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.70) < 1e-6

    def test_is_alias_for_accuracy(self):
        metrics = [(100, {"accuracy": 0.8})]
        assert weighted_average(metrics) == weighted_average_accuracy(metrics)
