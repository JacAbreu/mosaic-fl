import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.convergence import ConvergenceTracker
from experiments.experiment_server import CustomFedProxStrategy
from mosaicfl.core.model_v2 import SimplifiedBEHRT


class TestCustomFedProxStrategy:

    def _make_strategy(self, tmp_path):
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        with patch("experiments.experiment_server.FedProx.__init__", return_value=None):
            strategy = CustomFedProxStrategy.__new__(CustomFedProxStrategy)
            strategy.tracker = tracker
            strategy.history = history
            strategy.save_dir = str(tmp_path)
            strategy.on_converged = None
            strategy._round_counter = 0
            import os; os.makedirs(str(tmp_path), exist_ok=True)
        return strategy, tracker, history

    def test_aggregate_evaluate_populates_history(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.5, {"accuracy": 0.75})):
            strategy.aggregate_evaluate(1, [], [])
        assert 1 in history["rounds"]
        assert 0.75 in history["accuracy"]

    def test_aggregate_evaluate_raises_stop_on_convergence(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        tracker.history = [0.80, 0.800, 0.801]
        tracker.stable_count = 2
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.3, {"accuracy": 0.800})):
            with pytest.raises(StopIteration):
                strategy.aggregate_evaluate(4, [], [])

    def test_on_converged_callback_called(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        callback = MagicMock()
        strategy.on_converged = callback
        tracker.history = [0.80, 0.800, 0.801]
        tracker.stable_count = 2
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.3, {"accuracy": 0.800})):
            with pytest.raises(StopIteration):
                strategy.aggregate_evaluate(4, [], [])
        callback.assert_called_once()

    def test_aggregate_fit_saves_checkpoint(self, tmp_path):
        import flwr as fl
        strategy, tracker, history = self._make_strategy(tmp_path)
        real_model = SimplifiedBEHRT(use_cls_token=True)
        ndarrays = [v.numpy() for v in real_model.state_dict().values()]
        params = fl.common.ndarrays_to_parameters(ndarrays)
        with patch.object(type(strategy).__bases__[0], "aggregate_fit",
                          return_value=(params, {})):
            strategy.aggregate_fit(3, [], [])
        assert (tmp_path / "round_3.pt").exists()
        assert history["last_checkpoint"] is not None

    def test_history_grows_across_rounds(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        for i, acc in enumerate([0.70, 0.72, 0.74], 1):
            with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                               return_value=(0.5, {"accuracy": acc})):
                try:
                    strategy.aggregate_evaluate(i, [], [])
                except StopIteration:
                    break
        assert len(history["rounds"]) == len(history["accuracy"])
        assert len(history["rounds"]) >= 1
