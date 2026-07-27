"""
Testes para DataSource.find_vocab_candidates() — descoberta local de candidatos
de vocabulário (achado 2026-07-26, desenho do vocabulário federado bidirecional).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_client.datasource.base import DataSource
from infrastructure.mosaicfl_client.datasource.sgbd import SGBDDataSource
from infrastructure.mosaicfl_client.datasource.simulated import SimulatedDataSource
from infrastructure.mosaicfl_client.datasource.csv_source import CSVDataSource


class TestDataSourceDefault:
    def test_base_default_is_empty_list(self):
        class _Dummy(DataSource):
            def load(self, vocab=None): return None
            def get_metadata(self): return {}

        assert _Dummy().find_vocab_candidates({"PCR_HIGH": 1}) == []

    def test_simulated_source_default(self):
        assert SimulatedDataSource().find_vocab_candidates({}) == []

    def test_csv_source_default(self):
        assert CSVDataSource(filepath="fake.csv").find_vocab_candidates({}) == []


def _mock_engine_with_rows(rows):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


class TestSGBDFindVocabCandidates:
    def _source(self):
        return SGBDDataSource(connection_string="postgresql://fake", hospital_id="BPSP")

    def test_returns_empty_without_connection_or_hospital(self, monkeypatch):
        # connection_string="" cai no fallback de FL_DB_URL do ambiente (mesmo
        # comportamento de SGBDDataSource.__init__) — sem isolar isso, o teste
        # bateria no banco real se a variável estiver exportada na sessão do shell
        # (achado ao rodar este teste: FL_DB_URL estava exportado, teste consultou
        # dado de verdade sem querer). monkeypatch.delenv garante isolamento real.
        monkeypatch.delenv("FL_DB_URL", raising=False)
        assert SGBDDataSource(connection_string="", hospital_id="BPSP").find_vocab_candidates({}) == []
        assert SGBDDataSource(connection_string="postgresql://fake", hospital_id="").find_vocab_candidates({}) == []

    def test_excludes_analytes_already_in_vocab(self):
        rows = [
            SimpleNamespace(analyte="PCR", n_records=200, has_real_ref=False),
            SimpleNamespace(analyte="LINFOCITOS", n_records=300, has_real_ref=True),
        ]
        with patch("sqlalchemy.create_engine", return_value=_mock_engine_with_rows(rows)):
            candidates = self._source().find_vocab_candidates({"PCR": 1})
        analytes = [c["analyte"] for c in candidates]
        assert "PCR" not in analytes
        assert "LINFOCITOS" in analytes

    def test_recognizes_high_normal_low_suffix_as_known(self):
        rows = [SimpleNamespace(analyte="PCR", n_records=200, has_real_ref=True)]
        with patch("sqlalchemy.create_engine", return_value=_mock_engine_with_rows(rows)):
            candidates = self._source().find_vocab_candidates({"PCR_HIGH": 1, "PCR_NORMAL": 2, "PCR_LOW": 3})
        assert candidates == []

    def test_returns_n_records_and_has_real_ref(self):
        rows = [SimpleNamespace(analyte="NEUTROFILOS", n_records=254614, has_real_ref=True)]
        with patch("sqlalchemy.create_engine", return_value=_mock_engine_with_rows(rows)):
            candidates = self._source().find_vocab_candidates({})
        assert candidates == [{"analyte": "NEUTROFILOS", "n_records": 254614, "has_real_ref": True}]

    def test_db_error_returns_empty_list_not_exception(self):
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("conexão recusada")
        with patch("sqlalchemy.create_engine", return_value=engine):
            candidates = self._source().find_vocab_candidates({})
        assert candidates == []

    def test_logs_result_with_hospital_and_candidate_names(self, caplog):
        rows = [SimpleNamespace(analyte="NEUTROFILOS", n_records=254614, has_real_ref=True)]
        with patch("sqlalchemy.create_engine", return_value=_mock_engine_with_rows(rows)), \
             caplog.at_level("INFO", logger="infrastructure.mosaicfl_client.datasource.sgbd"):
            self._source().find_vocab_candidates({})

        assert any("find_vocab_candidates" in r.message for r in caplog.records)
        assert any("hospital=BPSP" in r.message for r in caplog.records)
        assert any("NEUTROFILOS" in r.message for r in caplog.records)
