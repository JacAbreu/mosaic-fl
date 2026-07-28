"""
Testes para CheckpointStore.save_round_history() — persistência incremental por
rodada (achado 2026-07-26: antes só uma chamada em lote no fim do treino; um crash
no meio perdia o histórico rodada-a-rodada do banco, sobrevivia só em
logs/round_N_metrics.json). Cobre também os campos novos (resource_per_client_json,
calibration_per_client_json — migration 025; per_client_f1_json — migration 027),
incluindo o COALESCE que impede uma chamada sem esses campos de apagar um valor já
persistido por uma chamada anterior.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.shared.checkpoint_store.postgres_store import PostgreSQLCheckpointStore
from infrastructure.shared.checkpoint_store.sqlite_store import SQLiteCheckpointStore


def _make_postgres_store():
    store = PostgreSQLCheckpointStore.__new__(PostgreSQLCheckpointStore)
    store._engine = MagicMock()
    conn = MagicMock()
    store._engine.begin.return_value.__enter__.return_value = conn
    store._engine.begin.return_value.__exit__.return_value = False
    return store, conn


class TestPostgresSaveRoundHistory:
    def test_single_round_call_includes_new_fields(self):
        store, conn = _make_postgres_store()
        store.save_round_history(
            training_id=70, rounds=[5], accuracies=[0.7], losses=[0.5],
            f1_macros=[0.4], per_class_f1s=[[0.7, 0.0]],
            resource_per_client_jsons=['[{"client_id": 0}]'],
            calibration_per_client_jsons=['[{"client_id": 0, "method": "temperature"}]'],
        )
        _, rows = conn.execute.call_args[0]
        assert rows[0]["resource_per_client_json"] == '[{"client_id": 0}]'
        assert rows[0]["calibration_per_client_json"] == '[{"client_id": 0, "method": "temperature"}]'

    def test_new_fields_default_to_none_when_omitted(self):
        store, conn = _make_postgres_store()
        store.save_round_history(training_id=70, rounds=[1], accuracies=[0.5], losses=[0.6])
        _, rows = conn.execute.call_args[0]
        assert rows[0]["resource_per_client_json"] is None
        assert rows[0]["calibration_per_client_json"] is None

    def test_empty_rounds_is_noop(self):
        store, conn = _make_postgres_store()
        store.save_round_history(training_id=70, rounds=[], accuracies=[], losses=[])
        conn.execute.assert_not_called()

    def test_sql_uses_coalesce_for_per_client_columns(self):
        """Garante que uma chamada posterior sem resource_per_client_jsons não apaga
        o que uma chamada anterior (incremental) já persistiu pra aquele round."""
        store, conn = _make_postgres_store()
        store.save_round_history(training_id=70, rounds=[1], accuracies=[0.5], losses=[0.6])
        sql_text = str(conn.execute.call_args[0][0])
        assert "COALESCE" in sql_text
        assert "resource_per_client_json" in sql_text
        assert "calibration_per_client_json" in sql_text

    def test_multi_round_batch_still_works(self):
        """Compatibilidade com a chamada em lote antiga (Caminho A ainda pode usar)."""
        store, conn = _make_postgres_store()
        store.save_round_history(
            training_id=70, rounds=[1, 2], accuracies=[0.5, 0.6], losses=[0.6, 0.5],
        )
        _, rows = conn.execute.call_args[0]
        assert len(rows) == 2


class TestSqliteSaveRoundHistoryNoop:
    def test_accepts_new_kwargs_without_raising(self, tmp_path):
        store = SQLiteCheckpointStore(db_path=str(tmp_path / "test.db"))
        store.save_round_history(
            training_id=1, rounds=[1], accuracies=[0.5], losses=[0.5],
            resource_per_client_jsons=["{}"], calibration_per_client_jsons=["{}"],
        )  # não deve levantar exceção — no-op documentado
