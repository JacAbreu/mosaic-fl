"""
Testes para a extração de per_class_f1 por cliente em ProductionFedProxStrategy.
aggregate_evaluate() — achado 2026-07-27: o F1 por classe de cada hospital,
antes da agregação federada, só existia no valor já combinado (média ponderada
nativa do Flower via per_class_f1_json em aggregated_metrics), nunca o valor
individual de cada cliente. Extraído de "results" (raw, per-cliente) e passado
pro save_round_history() incremental (ver migration 027). Mesmo padrão de
tests/unit/test_per_client_calibration_extraction.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(tmp_path):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy.global_model = SimplifiedBEHRT()
    strategy.vocab = {}
    strategy.tracker = ConvergenceTracker(threshold=0.005, patience=3)
    strategy.round_counter = 0
    strategy.should_stop = False
    strategy.on_round_complete = None
    strategy._state_store = None
    strategy._round_timer = None
    strategy._last_round_metrics = {}
    strategy._history_rounds = []
    strategy._history_accuracies = []
    strategy._history_losses = []
    strategy._history_f1_macros = []
    strategy._history_per_class_f1 = []
    strategy._gpu_energy_wh_total = 0.0
    strategy._gpu_power_samples = []
    strategy._peak_ram_mb = 0.0
    strategy._cpu_pct_samples = []
    strategy._last_round_resources_json = None
    strategy._training_id = 70
    strategy._checkpoint_store = MagicMock()
    strategy._num_rounds = 50
    strategy._training_completed = False
    strategy.CHECKPOINT_DIR = tmp_path / "checkpoints"
    strategy.LOG_DIR = tmp_path / "logs"
    strategy.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strategy.LOG_DIR.mkdir(parents=True, exist_ok=True)
    from infrastructure.mosaicfl_server.state_store import TrainingState
    strategy._current_state = TrainingState()
    return strategy


def _evaluate_result(client_id, per_class_f1=None):
    metrics = {"accuracy": 0.7, "client_id": client_id}
    if per_class_f1 is not None:
        import json
        metrics["per_class_f1_json"] = json.dumps(per_class_f1)
    return (None, SimpleNamespace(metrics=metrics))


class TestPerClientF1Extraction:
    def test_extracts_per_class_f1_per_client(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        results = [
            _evaluate_result(0, [0.8, 0.0, 0.9, 0.6, 0.5]),
            _evaluate_result(1, [0.1, 0.0, 0.05, 0.7, 0.4]),
        ]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        import json
        per_client = json.loads(kwargs["per_client_f1_jsons"][0])
        assert per_client == [
            {"client_id": 0, "per_class_f1": [0.8, 0.0, 0.9, 0.6, 0.5]},
            {"client_id": 1, "per_class_f1": [0.1, 0.0, 0.05, 0.7, 0.4]},
        ]

    def test_no_per_class_f1_data_produces_none(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        results = [_evaluate_result(0), _evaluate_result(1)]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        assert kwargs["per_client_f1_jsons"] == [None]

    def test_empty_results_list(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, [], [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        assert kwargs["per_client_f1_jsons"] == [None]

    def test_asymmetric_local_collapse_is_visible_per_client(self, tmp_path):
        """Caso motivador: um cliente pode ter F1 bom na sua classe dominante
        enquanto o outro tem F1=0 na mesma classe — isso fica invisível no
        per_class_f1 agregado (média), mas visível aqui."""
        strategy = _make_strategy(tmp_path)
        results = [
            _evaluate_result(0, [0.9, 0.0, 0.0, 0.3, 0.2]),  # curado_pronto forte (BPSP-like)
            _evaluate_result(1, [0.0, 0.0, 0.85, 0.2, 0.3]),  # melhora_pronto forte (HSL-like)
        ]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        import json
        per_client = json.loads(kwargs["per_client_f1_jsons"][0])
        assert per_client[0]["per_class_f1"][0] == 0.9  # BPSP prediz curado_pronto bem localmente
        assert per_client[1]["per_class_f1"][2] == 0.85  # HSL prediz melhora_pronto bem localmente
