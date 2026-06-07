import sys
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.data_loader import DatabaseDataSource


class TestDatabaseDataSource:
    """
    DatabaseDataSource conecta a SGBDs via SQLAlchemy.
    Todos os testes mockam o SQLAlchemy para não exigir banco real.
    """

    def _make_source(self, conn="postgresql://user:pass@host/db", query="SELECT * FROM t"):
        return DatabaseDataSource(connection_string=conn, query=query)

    # ── __init__ ──────────────────────────────────────────────────────────────

    def test_init_stores_connection_string(self):
        src = self._make_source(conn="sqlite:///test.db")
        assert src.connection_string == "sqlite:///test.db"

    def test_init_stores_query(self):
        src = self._make_source(query="SELECT id FROM pacientes")
        assert src.query == "SELECT id FROM pacientes"

    def test_init_engine_is_none(self):
        src = self._make_source()
        assert src._engine is None

    def test_init_empty_connection_string_uses_default(self, monkeypatch):
        import mosaicfl.v2.data_loader as dl
        monkeypatch.setattr(dl, "DEFAULT_CONNECTION_STRING", "sqlite:///env.db")
        src = dl.DatabaseDataSource(connection_string="")
        assert "sqlite" in src.connection_string

    # ── is_available() ────────────────────────────────────────────────────────

    def test_is_available_false_when_no_connection_string(self, monkeypatch):
        import mosaicfl.v2.data_loader as dl
        monkeypatch.setattr(dl, "DEFAULT_CONNECTION_STRING", "")
        src = DatabaseDataSource(connection_string="")
        assert src.is_available() is False

    def test_is_available_false_when_engine_raises(self):
        src = self._make_source()
        with patch("mosaicfl.v2.data_loader.DatabaseDataSource._get_engine",
                   side_effect=Exception("conexão recusada")):
            assert src.is_available() is False

    def test_is_available_true_when_connection_succeeds(self):
        src = self._make_source()
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        with patch("mosaicfl.v2.data_loader.DatabaseDataSource._get_engine",
                   return_value=mock_engine):
            assert src.is_available() is True

    # ── _get_engine() ─────────────────────────────────────────────────────────

    def test_get_engine_lazy_init(self):
        src = self._make_source()
        mock_engine = MagicMock()
        with patch("sqlalchemy.create_engine", return_value=mock_engine) as mock_create:
            e1 = src._get_engine()
            e2 = src._get_engine()
        mock_create.assert_called_once()
        assert e1 is e2

    def test_get_engine_raises_import_error_without_sqlalchemy(self):
        src = self._make_source()
        with patch.dict("sys.modules", {"sqlalchemy": None}):
            src._engine = None
            with pytest.raises(ImportError, match="SQLAlchemy"):
                src._get_engine()

    # ── load() ────────────────────────────────────────────────────────────────

    def test_load_raises_value_error_when_no_query(self, monkeypatch):
        import mosaicfl.v2.data_loader as dl
        monkeypatch.setattr(dl, "DEFAULT_QUERY", "")
        src = dl.DatabaseDataSource(connection_string="sqlite:///x.db", query="")
        with pytest.raises(ValueError, match="Query SQL"):
            src.load(query="")

    def test_load_raises_value_error_when_no_connection_string(self, monkeypatch):
        import mosaicfl.v2.data_loader as dl
        monkeypatch.setattr(dl, "DEFAULT_CONNECTION_STRING", "")
        src = dl.DatabaseDataSource(connection_string="", query="SELECT 1")
        with pytest.raises(ValueError, match="Connection string"):
            src.load()

    def test_load_returns_dataframe_from_sql(self):
        src = self._make_source()
        expected_df = pd.DataFrame({"id": [1, 2], "desfecho": [0, 1]})
        mock_engine = MagicMock()
        with patch("mosaicfl.v2.data_loader.DatabaseDataSource._get_engine",
                   return_value=mock_engine), \
             patch("pandas.read_sql", return_value=expected_df):
            df = src.load()
        assert len(df) == 2
        assert "id" in df.columns

    def test_load_raises_runtime_error_on_sql_failure(self):
        src = self._make_source()
        mock_engine = MagicMock()
        with patch("mosaicfl.v2.data_loader.DatabaseDataSource._get_engine",
                   return_value=mock_engine), \
             patch("pandas.read_sql", side_effect=Exception("tabela não existe")):
            with pytest.raises(RuntimeError, match="Erro ao executar query"):
                src.load()

    # ── _mask_connection_string() ─────────────────────────────────────────────

    def test_mask_hides_password(self):
        src = self._make_source(conn="postgresql://usuario:senha_secreta@host/db")
        masked = src._mask_connection_string()
        assert "senha_secreta" not in masked
        assert "***" in masked

    def test_mask_handles_no_password(self):
        src = self._make_source(conn="sqlite:///local.db")
        masked = src._mask_connection_string()
        assert "sqlite" in masked

    # ── list_tables() ─────────────────────────────────────────────────────────

    def test_list_tables_returns_empty_when_unavailable(self):
        src = self._make_source()
        with patch.object(src, "is_available", return_value=False):
            assert src.list_tables() == []

    def test_list_tables_returns_names_when_available(self):
        src = self._make_source()
        mock_engine = MagicMock()
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ["pacientes", "exames"]
        with patch.object(src, "is_available", return_value=True), \
             patch.object(src, "_get_engine", return_value=mock_engine), \
             patch("sqlalchemy.inspect", return_value=mock_inspector):
            tables = src.list_tables()
        assert "pacientes" in tables
        assert "exames" in tables

    def test_list_tables_returns_empty_on_inspect_error(self):
        src = self._make_source()
        with patch.object(src, "is_available", return_value=True), \
             patch.object(src, "_get_engine", return_value=MagicMock()), \
             patch("sqlalchemy.inspect", side_effect=Exception("inspect falhou")):
            assert src.list_tables() == []

    # ── _has_sqlalchemy() ─────────────────────────────────────────────────────

    def test_has_sqlalchemy_true_when_installed(self):
        src = self._make_source()
        assert src._has_sqlalchemy() is True

    def test_has_sqlalchemy_false_when_missing(self):
        src = self._make_source()
        with patch.dict("sys.modules", {"sqlalchemy": None}):
            assert src._has_sqlalchemy() is False
