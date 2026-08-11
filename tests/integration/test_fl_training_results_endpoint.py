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
        best_round=95, best_accuracy=0.75, converged=True, convergence_round=88, early_stop_enabled=False,
        macro_f1=0.42, macro_auc=0.83,
        ece=0.11, ece_pre=0.08, total_duration_s=15621.6, dp_noise_multiplier=None,
        dp_noise_strategy=None, dp_epsilon_simple=None, dp_epsilon_rdp=None,
        is_active_model=False, checkpoint_round=95,
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

    def test_checkpoint_mismatch_flagged(self, client_with_engine):
        """Achado 2026-07-28: fl_checkpoints.round divergindo de best_round é o
        bug do checkpoint nunca ser da melhor rodada (ver
        project_bug_checkpoint_nao_era_melhor_rodada). A tela precisa sinalizar
        isso, não só mostrar os dois números lado a lado sem destaque."""
        from infrastructure.mosaicfl_api import state
        state._db._engine = _mock_engine_connect(fetchall_result=[
            _training_row(id=77, best_round=95, checkpoint_round=110),
            _training_row(id=78, best_round=30, checkpoint_round=30),
        ])

        r = client_with_engine.get("/api/admin/fl-training-results", headers={"X-API-Key": "k"})
        data = r.json()["trainings"]
        assert data[0]["checkpoint_mismatch"] is True
        assert data[1]["checkpoint_mismatch"] is False

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
            first_results=[{"best_round": 95, "convergence_round": 88}],
        )

        r = client_with_engine.get("/api/admin/fl-training-results/77/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["best_round"] == 95
        assert data["convergence_round"] == 88
        assert data["rounds"][0]["per_class_f1"][1] == pytest.approx(0.17)
        assert data["rounds"][0]["per_client_f1"][0]["client_id"] == 1

    def test_returns_checkpoint_round_and_evaluation_json(self, client_with_engine):
        from infrastructure.mosaicfl_api import state
        round_row = SimpleNamespace(
            round=95, accuracy=0.75, f1_macro=0.42,
            per_class_f1=[0.84, 0.17, 0.0, 0.58, 0.50], per_client_f1_json=None,
        )
        conn = MagicMock()

        def _execute(stmt, params=None):
            result = MagicMock()
            sql = str(stmt)
            if "fl_trainings" in sql:
                result.mappings.return_value.first.return_value = {"best_round": 95, "convergence_round": 88}
            elif "fl_checkpoints" in sql:
                result.mappings.return_value.first.return_value = {
                    "round": 95, "evaluation_json": {"best_round": 95, "best_f1_macro": 0.42},
                }
            elif "fl_round_history" in sql:
                result.fetchall.return_value = [round_row]
            return result

        conn.execute.side_effect = _execute
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        r = client_with_engine.get("/api/admin/fl-training-results/77/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()
        assert data["checkpoint_round"] == 95
        assert data["evaluation_json"]["best_f1_macro"] == pytest.approx(0.42)

    def test_detects_repeated_per_class_f1_as_attractor_states(self, client_with_engine):
        """Achado 2026-08-01 (treino 85, DP uniforme): rodadas com per_class_f1
        IDÊNTICO indicam o modelo caindo repetidamente na mesma solução
        degenerada (só uma classe sobrevive), não convergência genuína — ver
        RepeatedStateGroup (schemas.py) e seção 16 do doc de pesquisa."""
        from infrastructure.mosaicfl_api import state
        rows = [
            SimpleNamespace(round=1, accuracy=0.07, f1_macro=0.03,
                             per_class_f1=[0.0, 0.0, 0.0, 0.0, 0.127081], per_client_f1_json=None),
            SimpleNamespace(round=4, accuracy=0.52, f1_macro=0.22,
                             per_class_f1=[0.722, 0.015, 0.027, 0.065, 0.270], per_client_f1_json=None),
            SimpleNamespace(round=60, accuracy=0.24, f1_macro=0.08,
                             per_class_f1=[0.0, 0.0, 0.0, 0.388281, 0.0], per_client_f1_json=None),
            SimpleNamespace(round=74, accuracy=0.07, f1_macro=0.03,
                             per_class_f1=[0.0, 0.0, 0.0, 0.0, 0.127081], per_client_f1_json=None),
            SimpleNamespace(round=75, accuracy=0.24, f1_macro=0.08,
                             per_class_f1=[0.0, 0.0, 0.0, 0.388281, 0.0], per_client_f1_json=None),
        ]
        state._db._engine = _mock_engine_connect(
            fetchall_result=rows,
            first_results=[{"best_round": 4, "convergence_round": 74}],
        )

        r = client_with_engine.get("/api/admin/fl-training-results/85/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()

        states = {tuple(s["rounds"]): s for s in data["repeated_states"]}
        assert (1, 74) in states
        assert states[(1, 74)]["dominant_class"] == "melhora_internado_grave"
        assert (60, 75) in states
        assert states[(60, 75)]["dominant_class"] == "melhora_internado_breve"
        # Round 4 (único, não-degenerado — várias classes com F1 > 0) não deve
        # aparecer como estado repetido.
        assert all(4 not in s["rounds"] for s in data["repeated_states"])

    def test_class_best_rounds_independent_of_overall_best_round(self, client_with_engine):
        """Achado 2026-08-02 — pedido explícito da autora: quer ver em qual
        rodada CADA classe teve seu melhor F1, mesmo que essa rodada não seja
        o best_round oficial (que usa f1_macro/accuracy agregado) nem a rodada
        de convergência. Classe 0 pico na 10, classe 3 pico na 20 — nenhuma das
        duas é o best_round=4 nem o convergence_round=30 deste cenário."""
        from infrastructure.mosaicfl_api import state
        rows = [
            SimpleNamespace(round=4, accuracy=0.50, f1_macro=0.30,
                             per_class_f1=[0.5, 0.1, 0.1, 0.3, 0.2], per_client_f1_json=None),
            SimpleNamespace(round=10, accuracy=0.55, f1_macro=0.28,
                             per_class_f1=[0.9, 0.0, 0.0, 0.1, 0.1], per_client_f1_json=None),
            SimpleNamespace(round=20, accuracy=0.40, f1_macro=0.25,
                             per_class_f1=[0.2, 0.0, 0.0, 0.8, 0.0], per_client_f1_json=None),
            SimpleNamespace(round=30, accuracy=0.45, f1_macro=0.20,
                             per_class_f1=[0.3, 0.0, 0.0, 0.2, 0.1], per_client_f1_json=None),
        ]
        state._db._engine = _mock_engine_connect(
            fetchall_result=rows,
            first_results=[{"best_round": 4, "convergence_round": 30}],
        )

        r = client_with_engine.get("/api/admin/fl-training-results/90/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()

        by_class = {c["class_name"]: c for c in data["class_best_rounds"]}
        assert by_class["curado_pronto"]["round"] == 10
        assert by_class["curado_pronto"]["f1"] == pytest.approx(0.9)
        assert by_class["melhora_internado_breve"]["round"] == 20
        assert by_class["melhora_internado_breve"]["f1"] == pytest.approx(0.8)

        assert data["best_round_detail"]["round"] == 4
        assert data["best_round_detail"]["f1_macro"] == pytest.approx(0.30)
        assert data["convergence_round_detail"]["round"] == 30
        assert data["convergence_round_detail"]["f1_macro"] == pytest.approx(0.20)

    def test_last_round_is_final_row_of_round_history_not_n_rounds_done(self, client_with_engine):
        """Pedido explícito da autora (2026-08-09): comparar F1 por classe entre
        melhor rodada, rodada de convergência e ÚLTIMA rodada — training_id=11
        (sem DP, sem early stop) mostrou 2 classes colapsando (F1=0) justamente
        na última rodada, sinal que best_round/convergence_round sozinhos não
        revelam. last_round vem da última linha real de fl_round_history, não
        de n_rounds_done (que fica em fl_trainings, não é consultado aqui)."""
        from infrastructure.mosaicfl_api import state
        rows = [
            SimpleNamespace(round=4, accuracy=0.50, f1_macro=0.30,
                             per_class_f1=[0.5, 0.1, 0.1, 0.3, 0.2], per_client_f1_json=None),
            SimpleNamespace(round=30, accuracy=0.45, f1_macro=0.20,
                             per_class_f1=[0.3, 0.0, 0.0, 0.2, 0.1], per_client_f1_json=None),
            SimpleNamespace(round=75, accuracy=0.75, f1_macro=0.39,
                             per_class_f1=[0.81, 0.0, 0.0, 0.70, 0.45], per_client_f1_json=None),
        ]
        state._db._engine = _mock_engine_connect(
            fetchall_result=rows,
            first_results=[{"best_round": 4, "convergence_round": 30}],
        )

        r = client_with_engine.get("/api/admin/fl-training-results/91/rounds", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        data = r.json()

        assert data["last_round"] == 75
        assert data["last_round_detail"]["f1_macro"] == pytest.approx(0.39)
        assert data["last_round_detail"]["per_class_f1"][1] == pytest.approx(0.0)
        assert data["last_round_detail"]["per_class_f1"][2] == pytest.approx(0.0)

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
    def test_queries_restrict_to_partition_mode_natural(self, client_with_engine):
        """Achado 2026-07-29: sem esse filtro, um treino do Caminho A com
        FL_PARTITION_MODE=iid_simulado (dado artificialmente balanceado, muito
        mais fácil de acertar macro_f1) podia vencer a comparação contra o
        non-IID real de BPSP/HSL — maçã com laranja."""
        from infrastructure.mosaicfl_api import state
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = conn
        state._db._engine = engine

        client_with_engine.get("/api/admin/fl-training-results/compare", headers={"X-API-Key": "k"})

        sql_calls = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert any("partition_mode = 'natural'" in sql for sql in sql_calls)
        assert len(sql_calls) == 2  # latest + best, ambas filtradas

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
            if "ORDER BY t.id DESC LIMIT 1" in sql:
                result.fetchone.return_value = latest
            elif "ORDER BY t.macro_f1 DESC LIMIT 1" in sql:
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


class TestPredictPage:
    def test_page_served(self, client_with_engine):
        r = client_with_engine.get("/predict")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
