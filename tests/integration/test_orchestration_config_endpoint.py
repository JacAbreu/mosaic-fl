"""
Testes para GET/POST /api/admin/orchestration-config — tela de peso de classe
explícito (cost-sensitive learning, Strategy pattern em mosaicfl.core.class_weighting).
Grava em clinical.fl_orchestration_config (migration 028), o mesmo canal que o
servidor de orquestração FL já usa pra empurrar proximal_mu/stop pro cliente a cada
rodada — ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _mock_engine_connect(overrides_row):
    conn = MagicMock()

    def _execute(stmt, params=None):
        result = MagicMock()
        result.mappings.return_value.first.return_value = overrides_row
        return result

    conn.execute.side_effect = _execute
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


def _mock_engine_begin():
    conn = MagicMock()
    engine = MagicMock()
    begin_ctx = MagicMock()
    begin_ctx.__enter__.return_value = conn
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx
    return engine, conn


@pytest.fixture()
def client_with_engine(monkeypatch):
    import infrastructure.mosaicfl_api.service as svc
    from infrastructure.mosaicfl_api import state
    from fastapi.testclient import TestClient

    mock_engine = MagicMock()
    mock_engine.checkpoint_path = None
    state._engine = mock_engine

    return TestClient(svc.app)


class TestGetOrchestrationConfig:
    def test_returns_empty_overrides_when_none_configured(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect({"class_weight_overrides_json": None})

        r = client_with_engine.get("/api/admin/orchestration-config", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["class_weight_overrides"] == {}
        assert "curado_internado" in data["class_labels"]
        assert data["class_weight_clamp"] == pytest.approx(15.0)

    def test_returns_configured_overrides(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(
            {"class_weight_overrides_json": {"curado_internado": 25.0}}
        )

        r = client_with_engine.get("/api/admin/orchestration-config", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert r.json()["class_weight_overrides"] == {"curado_internado": 25.0}

    def test_no_row_returns_empty_overrides(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(None)

        r = client_with_engine.get("/api/admin/orchestration-config", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert r.json()["class_weight_overrides"] == {}

    def test_requires_auth(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect({"class_weight_overrides_json": None})

        r = client_with_engine.get("/api/admin/orchestration-config")
        assert r.status_code == 403

    def test_db_error_returns_503_not_500(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("conexão recusada")
        state._db._engine = broken_engine

        r = client_with_engine.get("/api/admin/orchestration-config", headers={"X-API-Key": "k"})
        assert r.status_code == 503


class TestSetClassWeightOverrides:
    def test_saves_valid_overrides(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        engine, conn = _mock_engine_begin()
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {"curado_internado": 25.0}},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 200
        assert r.json()["class_weight_overrides"] == {"curado_internado": 25.0}

        call_args = conn.execute.call_args.args
        import json
        assert json.loads(call_args[1]["overrides"]) == {"curado_internado": 25.0}

    def test_empty_overrides_clears_column(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        engine, conn = _mock_engine_begin()
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {}},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 200
        call_args = conn.execute.call_args.args
        assert call_args[1]["overrides"] is None

    def test_unknown_class_name_returns_422(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        engine, _ = _mock_engine_begin()
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {"classe_que_nao_existe": 25.0}},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 422

    def test_non_positive_weight_returns_422(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        engine, _ = _mock_engine_begin()
        state._db._engine = engine

        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {"curado_internado": 0.0}},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 422

    def test_requires_auth(self, client_with_engine):
        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {"curado_internado": 25.0}},
        )
        assert r.status_code == 403

    def test_db_error_returns_503(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        broken_engine = MagicMock()
        broken_engine.begin.side_effect = RuntimeError("conexão recusada")
        state._db._engine = broken_engine

        r = client_with_engine.post(
            "/api/admin/orchestration-config/class-weights",
            json={"overrides": {"curado_internado": 25.0}},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 503


class TestClassWeightsPage:
    def test_page_served_at_class_weights_route(self, client_with_engine):
        r = client_with_engine.get("/class-weights")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
