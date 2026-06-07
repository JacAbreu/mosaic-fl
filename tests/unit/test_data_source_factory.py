import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.data_loader import DataSourceFactory, FileDataSource, DatabaseDataSource


class TestDataSourceFactory:

    # ── create() ──────────────────────────────────────────────────────────────

    def test_create_csv_returns_file_source(self):
        assert isinstance(DataSourceFactory.create("csv"), FileDataSource)

    def test_create_excel_returns_file_source(self):
        assert isinstance(DataSourceFactory.create("excel"), FileDataSource)

    def test_create_json_returns_file_source(self):
        assert isinstance(DataSourceFactory.create("json"), FileDataSource)

    def test_create_parquet_returns_file_source(self):
        assert isinstance(DataSourceFactory.create("parquet"), FileDataSource)

    def test_create_file_keyword_returns_file_source(self):
        assert isinstance(DataSourceFactory.create("file"), FileDataSource)

    def test_create_postgresql_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("postgresql"), DatabaseDataSource)

    def test_create_postgres_alias_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("postgres"), DatabaseDataSource)

    def test_create_mysql_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("mysql"), DatabaseDataSource)

    def test_create_sqlite_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("sqlite"), DatabaseDataSource)

    def test_create_mssql_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("mssql"), DatabaseDataSource)

    def test_create_database_keyword_returns_database_source(self):
        assert isinstance(DataSourceFactory.create("database"), DatabaseDataSource)

    def test_create_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="não suportada"):
            DataSourceFactory.create("redis")

    def test_create_case_insensitive(self):
        assert isinstance(DataSourceFactory.create("CSV"), FileDataSource)
        assert isinstance(DataSourceFactory.create("Json"), FileDataSource)

    def test_create_passes_kwargs_to_source(self, tmp_path):
        src = DataSourceFactory.create("csv", base_dir=tmp_path)
        assert isinstance(src, FileDataSource)

    # ── auto_detect() ─────────────────────────────────────────────────────────

    def test_auto_detect_uses_db_when_available(self):
        with patch("mosaicfl.core.data_loader.DatabaseDataSource.is_available",
                   return_value=True), \
             patch("mosaicfl.core.data_loader.DEFAULT_CONNECTION_STRING", "postgresql://x"):
            src = DataSourceFactory.auto_detect()
        assert isinstance(src, DatabaseDataSource)

    def test_auto_detect_falls_back_to_file(self):
        with patch("mosaicfl.core.data_loader.DEFAULT_CONNECTION_STRING", ""), \
             patch("mosaicfl.core.data_loader.FileDataSource.is_available",
                   return_value=True):
            src = DataSourceFactory.auto_detect()
        assert isinstance(src, FileDataSource)

    def test_auto_detect_raises_when_nothing_available(self):
        with patch("mosaicfl.core.data_loader.DEFAULT_CONNECTION_STRING", ""), \
             patch("mosaicfl.core.data_loader.FileDataSource.is_available",
                   return_value=False):
            with pytest.raises(RuntimeError, match="Nenhuma fonte"):
                DataSourceFactory.auto_detect()
