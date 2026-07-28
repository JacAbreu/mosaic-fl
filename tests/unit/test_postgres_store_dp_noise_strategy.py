"""
Testes para PostgreSQLCheckpointStore.update_evaluation_metrics() cobrindo
dp_noise_strategy/dp_noise_group_multipliers_json (migration 029) — auditoria de
qual estratégia de ruído DP (mosaicfl.core.dp_noise) foi usada por treinamento.
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


class TestUpdateEvaluationMetricsDpNoiseStrategy:
    def test_persists_strategy_and_group_multipliers(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(
            training_id=70,
            dp_noise_multiplier=1.0,
            dp_noise_strategy="layer_group",
            dp_noise_group_multipliers_json='{"head": 0.5, "transformer": 1.0}',
        )
        call_args = conn.execute.call_args.args
        params = call_args[1]
        assert params["dp_noise_strategy"] == "layer_group"
        assert params["dp_noise_group_multipliers_json"] == '{"head": 0.5, "transformer": 1.0}'

    def test_defaults_to_none_when_dp_disabled(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(training_id=70)
        call_args = conn.execute.call_args.args
        params = call_args[1]
        assert params["dp_noise_strategy"] is None
        assert params["dp_noise_group_multipliers_json"] is None

    def test_sql_casts_group_multipliers_to_jsonb(self):
        store, conn = _make_store()
        store.update_evaluation_metrics(training_id=70)
        sql_text = str(conn.execute.call_args.args[0])
        assert "cast(:dp_noise_group_multipliers_json as jsonb)" in sql_text
