import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.mosaicfl_server.config_loader import (
    ChromaDBConfigLoader,
    FileConfigLoader,
    PostgreSQLConfigLoader,
    get_config_loader,
)

_PG_URL = "postgresql://mosaicfl:x@localhost/mosaicfl"


class TestGetConfigLoader:

    def test_file_backend_returns_file_loader(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "file")
        assert isinstance(get_config_loader(), FileConfigLoader)

    def test_file_backend_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "FILE")
        assert isinstance(get_config_loader(), FileConfigLoader)

    def test_chroma_backend_returns_chroma_loader(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "chroma")
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        assert isinstance(get_config_loader(), ChromaDBConfigLoader)

    def test_postgres_backend_returns_postgres_loader(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "postgres")
        monkeypatch.setenv("FL_DB_URL", _PG_URL)
        with patch("sqlalchemy.create_engine"):
            loader = get_config_loader()
        assert isinstance(loader, PostgreSQLConfigLoader)

    def test_default_backend_is_postgres(self, monkeypatch):
        monkeypatch.delenv("FL_CONFIG_BACKEND", raising=False)
        monkeypatch.setenv("FL_DB_URL", _PG_URL)
        with patch("sqlalchemy.create_engine"):
            loader = get_config_loader()
        assert isinstance(loader, PostgreSQLConfigLoader)

    def test_unknown_backend_falls_back_to_postgres(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "redis")
        monkeypatch.setenv("FL_DB_URL", _PG_URL)
        with patch("sqlalchemy.create_engine"):
            loader = get_config_loader()
        assert isinstance(loader, PostgreSQLConfigLoader)
