import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.mosaicfl_server.config_loader import (
    ChromaDBConfigLoader,
    FileConfigLoader,
    get_config_loader,
)


class TestGetConfigLoader:

    def test_file_backend_returns_file_loader(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "file")
        loader = get_config_loader()
        assert isinstance(loader, FileConfigLoader)

    def test_file_backend_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "FILE")
        loader = get_config_loader()
        assert isinstance(loader, FileConfigLoader)

    def test_chroma_backend_returns_chroma_loader(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "chroma")
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)

    def test_default_backend_is_chroma(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FL_CONFIG_BACKEND", raising=False)
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)

    def test_unknown_backend_falls_back_to_chroma(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "redis")
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)
