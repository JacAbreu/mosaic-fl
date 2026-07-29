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

    def test_num_classes_valid(self):
        assert NUM_CLASSES >= 2

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
        cfg = ModelConfig(vocab_size=5000, num_classes=3, class_labels=("a", "b", "c"))
        assert cfg.vocab_size == 5000
        assert cfg.num_classes == 3
        assert len(cfg.class_labels) == 3

    def test_class_labels_length_matches_num_classes(self):
        assert len(NUM_CLASSES * ("x",)) == NUM_CLASSES  # tautologia para doc
        from mosaicfl.core.config import MODEL_CFG
        assert len(MODEL_CFG.class_labels) == MODEL_CFG.num_classes

    def test_class_labels_mismatch_raises(self):
        from mosaicfl.core.config import ModelConfig
        with pytest.raises(ValueError, match="FL_CLASS_LABELS"):
            ModelConfig(num_classes=3, class_labels=("a", "b"))

    def test_internado_breve_max_days_default_preserves_historical_behavior(self):
        from mosaicfl.core.config import MODEL_CFG
        assert MODEL_CFG.internado_breve_max_days == 10

    def test_internado_breve_max_days_reads_env(self, monkeypatch):
        from mosaicfl.core.config import ModelConfig
        monkeypatch.setenv("FL_INTERNADO_BREVE_MAX_DAYS", "5")
        assert ModelConfig().internado_breve_max_days == 5

    def test_demo_dim_default_preserves_historical_behavior(self):
        from mosaicfl.core.config import MODEL_CFG
        assert MODEL_CFG.demo_dim == 0

    def test_demo_dim_reads_env(self, monkeypatch):
        from mosaicfl.core.config import ModelConfig
        monkeypatch.setenv("FL_DEMO_DIM", "2")
        assert ModelConfig().demo_dim == 2
