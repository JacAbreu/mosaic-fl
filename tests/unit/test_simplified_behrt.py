import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.config import EMBED_DIM, NUM_CLASSES, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, MAX_SEQ_LEN
from mosaicfl.v2.model_v2 import SimplifiedBEHRT


class TestSimplifiedBEHRT:

    @pytest.fixture
    def model_v2(self):
        return SimplifiedBEHRT(use_cls_token=True)

    @pytest.fixture
    def model_no_cls(self):
        return SimplifiedBEHRT(use_cls_token=False)

    def test_forward_returns_correct_shape(self, model_v2):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        logits = model_v2(x)
        assert logits.shape == (4, NUM_CLASSES)

    def test_forward_with_attention(self, model_v2):
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        logits, attn = model_v2(x, return_attention=True)
        assert logits.shape == (2, NUM_CLASSES)
        assert attn.shape[0] == NUM_LAYERS
        assert attn.shape[1] == 2

    def test_cls_token_prepended(self, model_v2):
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        _, attn = model_v2(x, return_attention=True)
        assert attn.shape[3] == 17
        assert attn.shape[4] == 17

    def test_no_cls_token_mean_pooling(self, model_no_cls):
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        _, attn = model_no_cls(x, return_attention=True)
        assert attn.shape[3] == 16

    def test_masked_mean_pool_excludes_padding(self, model_v2):
        emb = torch.ones(1, 5, EMBED_DIM)
        emb[0, 3:] = 0.0
        mask = torch.tensor([[False, False, False, True, True]])
        pooled = model_v2._masked_mean_pool(emb, mask)
        expected = torch.ones(1, EMBED_DIM)
        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_masked_mean_pool_all_padding_no_nan(self, model_v2):
        emb = torch.randn(1, 4, EMBED_DIM)
        mask = torch.ones(1, 4, dtype=torch.bool)
        pooled = model_v2._masked_mean_pool(emb, mask)
        assert not torch.isnan(pooled).any()

    def test_deterministic_in_eval(self, model_v2):
        model_v2.eval()
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        with torch.no_grad():
            assert torch.allclose(model_v2(x), model_v2(x))

    def test_has_trainable_parameters(self, model_v2):
        params = [p for p in model_v2.parameters() if p.requires_grad]
        assert len(params) > 0
        assert sum(p.numel() for p in params) > 0

    def test_cls_token_is_parameter(self, model_v2):
        assert isinstance(model_v2.cls_token, torch.nn.Parameter)

    def test_pre_classifier_applied(self, model_v2):
        x = torch.randn(2, EMBED_DIM)
        out = model_v2.pre_classifier(x)
        assert out.shape == (2, EMBED_DIM)

    def test_state_dict_has_expected_keys(self, model_v2):
        keys = set(model_v2.state_dict().keys())
        assert "embedding.weight" in keys
        assert "cls_token" in keys
        assert "classifier.0.weight" in keys
