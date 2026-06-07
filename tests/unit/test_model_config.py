import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.config import VOCAB_SIZE, NUM_CLASSES, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MAX_SEQ_LEN


class TestModelConfig:

    def test_vocab_size_positive(self):
        assert VOCAB_SIZE > 0

    def test_embed_dim_positive(self):
        assert EMBED_DIM > 0

    def test_max_seq_len_power_of_two(self):
        assert MAX_SEQ_LEN > 0
        assert (MAX_SEQ_LEN & (MAX_SEQ_LEN - 1)) == 0

    def test_num_classes_binary(self):
        assert NUM_CLASSES == 2

    def test_num_layers_positive(self):
        assert NUM_LAYERS > 0

    def test_num_heads_divides_embed_dim(self):
        assert EMBED_DIM % NUM_HEADS == 0

    def test_frozen_raises_on_mutation(self):
        from mosaicfl.core.config import ModelConfig
        from dataclasses import FrozenInstanceError
        cfg = ModelConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.vocab_size = 999

    def test_custom_instance(self):
        from mosaicfl.core.config import ModelConfig
        cfg = ModelConfig(vocab_size=5000, num_classes=3)
        assert cfg.vocab_size == 5000
        assert cfg.num_classes == 3
