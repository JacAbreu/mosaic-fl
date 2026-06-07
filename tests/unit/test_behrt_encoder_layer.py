import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.config import EMBED_DIM, NUM_HEADS
from mosaicfl.core.model import BEHRTEncoderLayer


class TestBEHRTEncoderLayer:

    @pytest.fixture
    def layer(self):
        return BEHRTEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=128, dropout=0.0)

    def test_forward_shape(self, layer):
        x = torch.randn(2, 10, EMBED_DIM)
        assert layer(x).shape == (2, 10, EMBED_DIM)

    def test_forward_with_attention_shape(self, layer):
        x = torch.randn(2, 10, EMBED_DIM)
        out, attn = layer(x, return_attention=True)
        assert out.shape == (2, 10, EMBED_DIM)
        assert attn.shape == (2, NUM_HEADS, 10, 10)

    def test_padding_mask_applied(self, layer):
        x = torch.randn(1, 5, EMBED_DIM)
        mask_none = torch.zeros(1, 5, dtype=torch.bool)
        mask_last = torch.tensor([[False, False, False, True, True]])
        out_none = layer(x, src_key_padding_mask=mask_none)
        out_masked = layer(x, src_key_padding_mask=mask_last)
        assert not torch.allclose(out_none[0, :3], out_masked[0, :3], atol=1e-5)

    def test_deterministic_in_eval(self, layer):
        layer.eval()
        x = torch.randn(2, 8, EMBED_DIM)
        with torch.no_grad():
            assert torch.allclose(layer(x), layer(x))
