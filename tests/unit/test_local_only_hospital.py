"""
Testes para register_training(local_only_hospital=) — migration 030. Marca
treinos do Caminho B rodados com um único hospital conectado (min-clients=1),
baseline local comparável ao federado na mesma rede real.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.shared.checkpoint_store.postgres_store import PostgreSQLCheckpointStore
from infrastructure.shared.checkpoint_store.sqlite_store import SQLiteCheckpointStore


def _make_pg_store():
    store = PostgreSQLCheckpointStore.__new__(PostgreSQLCheckpointStore)
    store._engine = MagicMock()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1,)
    store._engine.begin.return_value.__enter__.return_value = conn
    store._engine.begin.return_value.__exit__.return_value = False
    return store, conn


class TestPostgresRegisterTrainingLocalOnly:
    def test_persists_local_only_hospital(self):
        store, conn = _make_pg_store()
        store.register_training(local_only_hospital="BPSP")
        params = conn.execute.call_args.args[1]
        assert params["local_only_hospital"] == "BPSP"

    def test_defaults_to_none_for_federated_training(self):
        store, conn = _make_pg_store()
        store.register_training()
        params = conn.execute.call_args.args[1]
        assert params["local_only_hospital"] is None

    def test_sql_includes_column(self):
        store, conn = _make_pg_store()
        store.register_training(local_only_hospital="HSL")
        sql_text = str(conn.execute.call_args.args[0])
        assert "local_only_hospital" in sql_text


class TestSqliteRegisterTrainingLocalOnly:
    def test_accepts_param_without_crashing(self, tmp_path):
        store = SQLiteCheckpointStore(db_path=str(tmp_path / "test.db"))
        training_id = store.register_training(local_only_hospital="BPSP")
        assert isinstance(training_id, int)
