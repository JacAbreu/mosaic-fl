import sys
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.config import VOCAB_SIZE, NUM_CLASSES
from mosaicfl.v2.model_v2 import SimplifiedBEHRT
from mosaicfl.v2.server_v2 import get_evaluate_fn


class TestGetEvaluateFn:

    def test_returns_callable(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        assert callable(fn)

    def test_returns_float_loss_and_metrics(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        loss, metrics = fn(1, params, {})
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_accuracy_bounded(self):
        x = torch.randint(1, VOCAB_SIZE, (16, 16))
        y = torch.zeros(16, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        _, metrics = fn(1, params, {})
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_rodada_in_metrics(self):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        _, metrics = fn(5, params, {})
        assert metrics["rodada"] == 5
