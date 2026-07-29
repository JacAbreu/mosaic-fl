"""
Testes para PostgreSQLCheckpointStore.update_evaluation_metrics() cobrindo
calibration_method_requested (migration 034) — o método PEDIDO (FED_CFG.calibration_method
visto pelo ServerApp), distinto do método ESCOLHIDO por cada cliente sob "auto"
(fl_round_history.calibration_per_client_json). Achado 2026-07-29: sem essa coluna, um
treino que rodou com "temperature" forçado (bug de roteamento de env var corrigido no
mesmo dia) era indistinguível, no banco, de um treino "auto" que escolheu temperature.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.shared.checkpoint_store.postgres_store import PostgreSQLCheckpointStore


def _make_store():
    store = PostgreSQLCheckpointStore.__new__(PostgreSQLCheckpointStore)
    store._engine = MagicMock()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__.return_value = conn
    store._engine.begin.return_value.__exit__.return_value = False
    return store, conn


class TestUpdateEvaluationMetricsCalibrationMethodRequested:
    def test_persists_requested_method(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(training_id=70, calibration_method_requested="auto")
        call_args = conn.execute.call_args.args
        params = call_args[1]
        assert params["calibration_method_requested"] == "auto"

    def test_defaults_to_none(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(training_id=70)
        call_args = conn.execute.call_args.args
        params = call_args[1]
        assert params["calibration_method_requested"] is None

    def test_sql_includes_column(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(training_id=70)
        sql_text = str(conn.execute.call_args.args[0])
        assert "calibration_method_requested=:calibration_method_requested" in sql_text
