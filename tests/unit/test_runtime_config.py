import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.config import DEVICE


class TestRuntimeConfig:

    def test_device_is_torch_device(self):
        import torch
        from mosaicfl.v2.config import RUNTIME_CFG
        assert isinstance(RUNTIME_CFG.device, torch.device)

    def test_device_defaults_to_cpu(self):
        assert str(DEVICE) == "cpu"

    def test_data_path_is_path(self):
        from mosaicfl.v2.config import RUNTIME_CFG
        assert isinstance(RUNTIME_CFG.data_path, Path)

    def test_chroma_path_is_path(self):
        from mosaicfl.v2.config import RUNTIME_CFG
        assert isinstance(RUNTIME_CFG.chroma_path, Path)

    def test_not_frozen_allows_mutation(self):
        from mosaicfl.v2.config import RuntimeConfig
        cfg = RuntimeConfig()
        cfg.embedding_model = "outro-modelo"
        assert cfg.embedding_model == "outro-modelo"

    def test_device_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("FL_DEVICE", "cpu")
        from mosaicfl.v2.config import RuntimeConfig
        cfg = RuntimeConfig()
        assert str(cfg.device) == "cpu"

    def test_use_ray_false_by_default(self):
        from mosaicfl.v2.config import RUNTIME_CFG
        assert RUNTIME_CFG.use_ray is False
