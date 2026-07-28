"""
Testes para GET /api/admin/fl-training-results[/{id}/rounds|/compare] — tela de
avaliação de treinamentos (/fl-training-results), resumo + per_class_f1/
per_client_f1 por rodada + comparação mais-recente-vs-melhor. Mesmo padrão de
engine mockado de test_orchestration_config_endpoint.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _training_row(**kwargs):
    defaults = dict(
        id=1, algorithm="FedProx", run_classification="ajuste", partition_mode="natural",
        status="completed", started_at=None, completed_at=None, n_rounds_done=110,
        best_round=95, best_accuracy=0.75, converged=True, macro_f1=0.42, macro_auc=0.83,
        ece=0.11, ece_pre=0.08, total_duration_s=15621.6, dp_noise_multiplier=None,
        dp_noise_strategy=None, is_active_model=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mock_engine_connect(fetchall_result=None, fetchone_result=None, first_results=None):
    conn = MagicMock()

    def _execute(stmt, params=None):
        result = MagicMock()
        result.fetchall.return_value = fetchall_result or []
        result.fetchone.return_value = fetchone_result
        if first_results is not None:
            result.mappings.return_value.first.return_value = first_results.pop(0) if first_results else None
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


class TestListTrainingResults:
    def test_returns_summary_list(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[_training_row(id=77), _training_row(id=76)])

        r = client_with_engine.get("/api/admin/fl-training-results", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()["trainings"]
        assert len(data) == 2
        assert data[0]["id"] == 77
        assert data[0]["macro_f1"] == pytest.approx(0.42)

    def test_empty_list(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/fl-training-results", headers={"X-API-Key": "k"})
        assert r.json()["trainings"] == []

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/fl-training-results")
        assert r.status_code == 403

    def test_db_error_returns_503(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        broken = MagicMock()
        broken.connect.side_effect = RuntimeError("conexão recusada")
        state._db._engine = broken

        r = client_with_engine.get("/api/admin/fl-training-results", headers={"X-API-Key": "k"})
        assert r.status_code == 503


class TestGetTrainingRounds:
    def test_returns_rounds_with_per_class_and_per_client_f1(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        round_row = SimpleNamespace(
            round=95, accuracy=0.75, f1_macro=0.42,
            per_class_f1=[0.84, 0.17, 0.0, 0.58, 0.50],
            per_client_f1_json=[{"client_id": 1, "per_class_f1": [0.9, 0.19, 0.0, 0.57, 0.53]}],
        )
        state._db._engine = _mock_engine_connect(
            fetchall_result=[round_row],
            first_results=[{"best_round": 95}],
        )

        r = client_with_engine.get("/api/admin/fl-training-results/77/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["best_round"] == 95
        assert data["rounds"][0]["per_class_f1"][1] == pytest.approx(0.17)
        assert data["rounds"][0]["per_client_f1"][0]["client_id"] == 1

    def test_unknown_training_id_returns_404(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[], first_results=[None])

        r = client_with_engine.get("/api/admin/fl-training-results/9999/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 404

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[])

        r = client_with_engine.get("/api/admin/fl-training-results/77/rounds")
        assert r.status_code == 403


class TestCompareTrainingResults:
    def test_compares_latest_and_best(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        latest = _training_row(id=77, macro_f1=0.42, best_round=95)
        best = _training_row(id=74, macro_f1=0.45, best_round=69)

        conn = MagicMock()
        call_count = {"n": 0}

        def _execute(stmt, params=None):
            call_count["n"] += 1
            result = MagicMock()
            sql = str(stmt)
            if "ORDER BY id DESC LIMIT 1" in sql:
                result.fetchone.return_value = latest
            elif "ORDER BY macro_f1 DESC LIMIT 1" in sql:
                result.fetchone.return_value = best
            elif "fl_round_history" in sql:
                pcf = [0.84, 0.17, 0.0, 0.58, 0.50] if params.get("tid") == 77 else [0.85, 0.0, 0.0, 0.68, 0.53]
                result.mappings.return_value.first.return_value = {"per_class_f1": pcf}
            return result

        conn.execute.side_effect = _execute
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        r = client_with_engine.get("/api/admin/fl-training-results/compare", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["latest"]["training"]["id"] == 77
        assert data["best"]["training"]["id"] == 74
        assert data["latest"]["per_class_f1"][1] == pytest.approx(0.17)
        assert data["best"]["per_class_f1"][1] == pytest.approx(0.0)

    def test_no_trainings_returns_none_sides(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        r = client_with_engine.get("/api/admin/fl-training-results/compare", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["latest"] is None
        assert data["best"] is None

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect()

        r = client_with_engine.get("/api/admin/fl-training-results/compare")
        assert r.status_code == 403


class TestFlTrainingResultsPage:
    def test_page_served(self, client_with_engine):
        r = client_with_engine.get("/fl-training-results")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
