"""
Testes para register_training(early_stop_enabled=) — migration 036. FED_CFG.early_stop
nunca era persistido (só aparecia no log do SuperLink); sem isso, n_rounds_done <
n_rounds_max é ambíguo (early stop de verdade vs. interrupção por outro motivo —
já se confundiu nesta mesma fase, training_id 3/4, ver linha do tempo).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.shared.checkpoint_store.postgres_store import PostgreSQLCheckpointStore


def _make_pg_store():
    store = PostgreSQLCheckpointStore.__new__(PostgreSQLCheckpointStore)
    store._engine = MagicMock()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1,)
    store._engine.begin.return_value.__enter__.return_value = conn
    store._engine.begin.return_value.__exit__.return_value = False
    return store, conn


class TestPostgresRegisterTrainingEarlyStop:
    def test_persists_early_stop_enabled_true(self):
        store, conn = _make_pg_store()
        store.register_training(early_stop_enabled=True)
        params = conn.execute.call_args.args[1]
        assert params["early_stop_enabled"] is True

    def test_defaults_to_false(self):
        store, conn = _make_pg_store()
        store.register_training()
        params = conn.execute.call_args.args[1]
        assert params["early_stop_enabled"] is False
