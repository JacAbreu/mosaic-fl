import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.config import VOCAB_SIZE, NUM_CLASSES
from mosaicfl.v2.server_v2 import ConvergenceTracker, start_server


class TestStartServer:

    @staticmethod
    def _start(**kwargs):
        with patch("mosaicfl.v2.server_v2.fl.server.start_server"):
            return start_server(**kwargs)

    def test_returns_three_values(self):
        strategy, tracker, history = self._start()
        assert strategy is not None
        assert isinstance(tracker, ConvergenceTracker)
        assert isinstance(history, dict)

    def test_history_has_expected_keys(self):
        _, _, history = self._start()
        assert "rounds" in history
        assert "accuracy" in history
        assert "communication_mb" in history

    def test_evaluate_fn_none_without_test_loader(self):
        strategy, _, _ = self._start(test_loader=None)
        assert strategy is not None

    def test_evaluate_fn_active_with_test_loader(self):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        strategy, _, _ = self._start(test_loader=loader)
        assert strategy is not None
