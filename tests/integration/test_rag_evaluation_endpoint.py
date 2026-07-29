"""
Testes para GET/POST /api/admin/rag-evaluations[/{id}/score|/report] — tela de
avaliação do RAG (/rag-evaluation). Mesmo padrão de engine mockado de
test_fl_training_results_endpoint.py. A nota HUMANA (likert_score) nunca é
inferida — só persistida via POST explícito; llm_judge_score é sempre
tratada como métrica separada e complementar (achado 2026-07-29).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _eval_row(**kwargs):
    from datetime import datetime, timezone
    defaults = dict(
        id=1, patient_id_hash="abc123hash", predicted_label="curado_pronto",
        risk_score=0.42, justificativa="Paciente com PCR baixo, evolução favorável.",
        fontes_json=[{"metadata": {"desfecho": "curado_pronto"}}],
        llm_backend="ollama", llm_model_used="gemma3:4b", llm_was_fallback=False,
        alucinacao_detectada=False, confiavel=True, likert_score=None,
        llm_judge_score=4, llm_judge_rationale="Coerente com os casos.",
        evaluator=None, checkpoint_round=95, created_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mock_engine_connect(fetchall_result=None):
    conn = MagicMock()

    def _execute(stmt, params=None):
        result = MagicMock()
        result.fetchall.return_value = fetchall_result or []
        return result

    conn.execute.side_effect = _execute
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


@pytest.fixture()
def client_with_engine(monkeypatch):
    import infrastructure.mosaicfl_api.service as svc
    from infrastructure.mosaicfl_api import state
    from fastapi.testclient import TestClient

    mock_engine = MagicMock()
    mock_engine.checkpoint_path = None
    state._engine = mock_engine

    return TestClient(svc.app)


class TestListRagEvaluations:
    def test_returns_pending_by_default(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[_eval_row(id=1), _eval_row(id=2)])

        r = client_with_engine.get("/api/admin/rag-evaluations", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()["evaluations"]
        assert len(data) == 2
        assert data[0]["likert_score"] is None
        assert data[0]["llm_judge_score"] == 4

    def test_pending_only_flag_in_query(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        conn = MagicMock()
        captured_sql = {}

        def _execute(stmt, params=None):
            captured_sql["sql"] = str(stmt)
            result = MagicMock()
            result.fetchall.return_value = []
            return result

        conn.execute.side_effect = _execute
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        client_with_engine.get("/api/admin/rag-evaluations?pending_only=false", headers={"X-API-Key": "k"})
        assert "likert_score IS NULL" not in captured_sql["sql"]

        client_with_engine.get("/api/admin/rag-evaluations?pending_only=true", headers={"X-API-Key": "k"})
        assert "likert_score IS NULL" in captured_sql["sql"]

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/rag-evaluations")
        assert r.status_code == 403

    def test_db_error_returns_503(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        broken = MagicMock()
        broken.connect.side_effect = RuntimeError("conexão recusada")
        state._db._engine = broken

        r = client_with_engine.get("/api/admin/rag-evaluations", headers={"X-API-Key": "k"})
        assert r.status_code == 503


class TestScoreRagEvaluation:
    def test_persists_human_score(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        conn = MagicMock()
        conn.execute.return_value.rowcount = 1
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/rag-evaluations/1/score",
            headers={"X-API-Key": "k"},
            json={"likert_score": 5, "evaluator": "Jacqueline"},
        )
        assert r.status_code == 200
        assert r.json()["likert_score"] == 5
        params = conn.execute.call_args.args[1]
        assert params["score"] == 5
        assert params["evaluator"] == "Jacqueline"

    def test_score_out_of_range_rejected(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect()

        r = client_with_engine.post(
            "/api/admin/rag-evaluations/1/score",
            headers={"X-API-Key": "k"},
            json={"likert_score": 9, "evaluator": "Jacqueline"},
        )
        assert r.status_code == 422

    def test_unknown_id_returns_404(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        conn = MagicMock()
        conn.execute.return_value.rowcount = 0
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/rag-evaluations/9999/score",
            headers={"X-API-Key": "k"},
            json={"likert_score": 3, "evaluator": "Jacqueline"},
        )
        assert r.status_code == 404

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect()

        r = client_with_engine.post(
            "/api/admin/rag-evaluations/1/score", json={"likert_score": 3, "evaluator": "x"},
        )
        assert r.status_code == 403


class TestRagEvaluationsReport:
    def test_reports_human_and_judge_side_by_side(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        rows = [
            _eval_row(id=1, likert_score=5, llm_judge_score=5, alucinacao_detectada=False, confiavel=True),
            _eval_row(id=2, likert_score=2, llm_judge_score=4, alucinacao_detectada=True, confiavel=False),
            _eval_row(id=3, likert_score=None, llm_judge_score=3, alucinacao_detectada=False, confiavel=True),
        ]
        state._db._engine = _mock_engine_connect(fetchall_result=rows)

        r = client_with_engine.get("/api/admin/rag-evaluations/report", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        d = r.json()
        assert d["n_total"] == 3
        assert d["n_human_scored"] == 2
        assert d["n_judge_scored"] == 3
        assert d["human_pct_ge4"] == pytest.approx(50.0)  # 1 de 2 (nota 5) >= 4
        assert d["n_both_scored"] == 2

    def test_empty_returns_zeros_without_crashing(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/rag-evaluations/report", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert r.json()["n_total"] == 0

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/rag-evaluations/report")
        assert r.status_code == 403


class TestRagEvaluationPage:
    def test_page_served(self, client_with_engine):
        r = client_with_engine.get("/rag-evaluation")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
