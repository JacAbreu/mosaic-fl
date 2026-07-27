"""
Testes para GET/POST /api/admin/vocab-anomalies — página de revisão de analitos
fora de padrão no catálogo (knowledge.term_dictionary), achado 2026-07-26 durante
a validação real do vocabulário federado bidirecional (canonical="183", puramente
numérico, aprovado automaticamente pela descoberta).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _mock_engine(term_dict_rows, count_by_canonical=None):
    count_by_canonical = count_by_canonical or {}
    conn = MagicMock()

    def _execute(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "FROM knowledge.term_dictionary td" in sql:
            result.fetchall.return_value = term_dict_rows
        elif "FROM metrics.exam_records" in sql:
            canonical = (params or {}).get("c")
            result.scalar.return_value = count_by_canonical.get(canonical, 0)
        elif sql.strip().startswith("UPDATE knowledge.term_dictionary"):
            canonical = (params or {}).get("canonical")
            result.rowcount = 1 if canonical in count_by_canonical or canonical == "183" else 0
        return result

    conn.execute.side_effect = _execute
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


def _row(canonical, source="AUTO_DISCOVERED", active=True, created_at=None,
         ref_low=None, ref_high=None, n_hospitals=None):
    return SimpleNamespace(
        canonical=canonical, source=source, active=active, created_at=created_at,
        ref_low=ref_low, ref_high=ref_high, n_hospitals=n_hospitals,
    )


@pytest.fixture()
def client_with_engine(monkeypatch):
    import infrastructure.mosaicfl_api.service as svc
    from infrastructure.mosaicfl_api import state
    from fastapi.testclient import TestClient

    mock_engine = MagicMock()
    mock_engine.checkpoint_path = None
    state._engine = mock_engine

    def _set_rows(rows, count_by_canonical=None):
        state._db._engine = _mock_engine(rows, count_by_canonical)

    return TestClient(svc.app), _set_rows


class TestListVocabAnomalies:
    def test_flags_purely_numeric_canonical(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([_row("183", ref_low=28.0, ref_high=100.0, n_hospitals=1)],
                 count_by_canonical={"183": 0})

        r = client.get("/api/admin/vocab-anomalies", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()["anomalies"]
        assert len(data) == 1
        assert data[0]["canonical"] == "183"
        assert data[0]["local_record_count"] == 0
        assert data[0]["n_hospitals"] == 1

    def test_excludes_normal_looking_canonicals(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([_row("LINFOCITOS"), _row("183")], count_by_canonical={"183": 5})

        r = client.get("/api/admin/vocab-anomalies", headers={"X-API-Key": "k"})
        data = r.json()["anomalies"]
        assert [a["canonical"] for a in data] == ["183"]

    def test_empty_catalog_returns_empty_list(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([])

        r = client.get("/api/admin/vocab-anomalies", headers={"X-API-Key": "k"})
        assert r.json()["anomalies"] == []

    def test_requires_auth(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([])
        r = client.get("/api/admin/vocab-anomalies")
        assert r.status_code == 403

    def test_db_error_returns_503_not_500(self, client_with_engine):
        client, set_rows = client_with_engine
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("conexão recusada")
        from infrastructure.mosaicfl_api import state
        state._db._engine = broken_engine

        r = client.get("/api/admin/vocab-anomalies", headers={"X-API-Key": "k"})
        assert r.status_code == 503


class TestCorrectVocabAnomaly:
    def test_deactivate_returns_200(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([_row("183")], count_by_canonical={"183": 0})

        r = client.post("/api/admin/vocab-anomalies/183", json={"active": False},
                        headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert r.json() == {"canonical": "183", "active": False}

    def test_reactivate_returns_200(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([_row("183")], count_by_canonical={"183": 0})

        r = client.post("/api/admin/vocab-anomalies/183", json={"active": True},
                        headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert r.json()["active"] is True

    def test_unknown_canonical_returns_404(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([], count_by_canonical={})

        r = client.post("/api/admin/vocab-anomalies/NAO_EXISTE", json={"active": False},
                        headers={"X-API-Key": "k"})
        assert r.status_code == 404

    def test_requires_auth(self, client_with_engine):
        client, set_rows = client_with_engine
        set_rows([_row("183")], count_by_canonical={"183": 0})
        r = client.post("/api/admin/vocab-anomalies/183", json={"active": False})
        assert r.status_code == 403


def _mock_engine_begin(target_exists, exam_records_updated=0):
    conn = MagicMock()

    def _execute(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if sql.strip().startswith("SELECT 1 FROM knowledge.term_dictionary"):
            result.fetchone.return_value = (1,) if target_exists else None
        elif "UPDATE metrics.exam_records" in sql:
            result.rowcount = exam_records_updated
        return result

    conn.execute.side_effect = _execute
    engine = MagicMock()
    begin_ctx = MagicMock()
    begin_ctx.__enter__.return_value = conn
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx
    return engine, conn


class TestRenameVocabAnomaly:
    def test_rename_when_target_does_not_exist(self, client_with_engine):
        client, _ = client_with_engine
        from infrastructure.mosaicfl_api import state
        engine, conn = _mock_engine_begin(target_exists=False, exam_records_updated=3)
        state._db._engine = engine

        r = client.post("/api/admin/vocab-anomalies/183/correct",
                        json={"correct_canonical": "AMILASE_NOVA"}, headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data == {
            "old_canonical": "183", "new_canonical": "AMILASE_NOVA",
            "merged": False, "exam_records_updated": 3,
        }
        executed_sql = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert any("UPDATE knowledge.term_dictionary SET canonical" in s for s in executed_sql)
        assert any("UPDATE knowledge.analyte_references SET canonical" in s for s in executed_sql)
        assert not any("DELETE FROM knowledge.term_dictionary" in s for s in executed_sql)

    def test_merge_when_target_already_exists(self, client_with_engine):
        client, _ = client_with_engine
        from infrastructure.mosaicfl_api import state
        engine, conn = _mock_engine_begin(target_exists=True, exam_records_updated=2133)
        state._db._engine = engine

        r = client.post("/api/admin/vocab-anomalies/183/correct",
                        json={"correct_canonical": "amilase"}, headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["new_canonical"] == "AMILASE"
        assert data["merged"] is True
        assert data["exam_records_updated"] == 2133
        executed_sql = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert any("DELETE FROM knowledge.term_dictionary" in s for s in executed_sql)
        assert any("INSERT INTO knowledge.term_dictionary" in s for s in executed_sql)
        assert any("DELETE FROM knowledge.analyte_references" in s for s in executed_sql)

    def test_rejects_empty_correct_canonical(self, client_with_engine):
        client, _ = client_with_engine
        from infrastructure.mosaicfl_api import state
        engine, _ = _mock_engine_begin(target_exists=False)
        state._db._engine = engine

        r = client.post("/api/admin/vocab-anomalies/183/correct",
                        json={"correct_canonical": "   "}, headers={"X-API-Key": "k"})
        assert r.status_code == 422

    def test_rejects_same_as_current(self, client_with_engine):
        client, _ = client_with_engine
        from infrastructure.mosaicfl_api import state
        engine, _ = _mock_engine_begin(target_exists=False)
        state._db._engine = engine

        r = client.post("/api/admin/vocab-anomalies/183/correct",
                        json={"correct_canonical": "183"}, headers={"X-API-Key": "k"})
        assert r.status_code == 422

    def test_requires_auth(self, client_with_engine):
        client, _ = client_with_engine
        r = client.post("/api/admin/vocab-anomalies/183/correct", json={"correct_canonical": "AMILASE"})
        assert r.status_code == 403

    def test_db_error_returns_503(self, client_with_engine):
        client, _ = client_with_engine
        from infrastructure.mosaicfl_api import state
        broken_engine = MagicMock()
        broken_engine.begin.side_effect = RuntimeError("conexão recusada")
        state._db._engine = broken_engine

        r = client.post("/api/admin/vocab-anomalies/183/correct",
                        json={"correct_canonical": "AMILASE"}, headers={"X-API-Key": "k"})
        assert r.status_code == 503
