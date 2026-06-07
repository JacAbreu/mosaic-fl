import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.config import EMBED_DIM, MAX_SEQ_LEN
from mosaicfl.v2.model_v2 import PositionalEncoding


class TestPositionalEncoding:

    def test_output_shape_preserved(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.randn(2, 10, 64)
        assert pe(x).shape == x.shape

    def test_modifies_zero_input(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.zeros(1, 5, 64)
        assert not torch.allclose(pe(x), x)

    def test_max_len_covers_cls_token(self):
        pe = PositionalEncoding(d_model=EMBED_DIM, max_len=MAX_SEQ_LEN + 1)
        x = torch.randn(2, MAX_SEQ_LEN + 1, EMBED_DIM)
        assert pe(x).shape == (2, MAX_SEQ_LEN + 1, EMBED_DIM)

    def test_different_positions_get_different_encoding(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])
