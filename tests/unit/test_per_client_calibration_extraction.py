"""
Testes para a extração de calibração por cliente em ProductionFedProxStrategy.
aggregate_evaluate() — achado 2026-07-26: o T/isotônico individual de cada
hospital, antes da agregação federada, só existia no log do próprio cliente
(local_calibration_fit), nunca persistido. Extraído de "results" (raw, per-cliente,
disponível antes de aggregated_metrics já vir combinado) e passado pro
save_round_history() incremental (ver migration 025).
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

    class _CurrentStateStub:
        run_id = None
        last_round = 0

    import infrastructure.mosaicfl_server.strategy.core as core_module
    return strategy, core_module


def _evaluate_result(client_id, calibration_method=None, temperature=None):
    metrics = {"accuracy": 0.7, "client_id": client_id}
    if calibration_method:
        metrics["calibration_method"] = calibration_method
        if temperature is not None:
            metrics["temperature"] = temperature
    return (None, SimpleNamespace(metrics=metrics))


class TestPerClientCalibrationExtraction:
    def test_extracts_temperature_per_client(self, tmp_path):
        strategy, core_module = _make_strategy(tmp_path)
        results = [
            _evaluate_result(0, "temperature", 1.31),
            _evaluate_result(1, "temperature", 1.85),
        ]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        import json
        per_client = json.loads(kwargs["calibration_per_client_jsons"][0])
        assert per_client == [
            {"client_id": 0, "method": "temperature", "temperature": 1.31},
            {"client_id": 1, "method": "temperature", "temperature": 1.85},
        ]

    def test_no_calibration_data_produces_none(self, tmp_path):
        strategy, core_module = _make_strategy(tmp_path)
        results = [_evaluate_result(0), _evaluate_result(1)]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        assert kwargs["calibration_per_client_jsons"] == [None]

    def test_isotonic_method_without_temperature_field(self, tmp_path):
        strategy, core_module = _make_strategy(tmp_path)
        results = [_evaluate_result(0, "isotonic")]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        import json
        per_client = json.loads(kwargs["calibration_per_client_jsons"][0])
        assert per_client == [{"client_id": 0, "method": "isotonic"}]

    def test_empty_results_list(self, tmp_path):
        strategy, core_module = _make_strategy(tmp_path)
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, [], [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        assert kwargs["calibration_per_client_jsons"] == [None]

    def test_incremental_save_called_with_current_round_only(self, tmp_path):
        strategy, core_module = _make_strategy(tmp_path)
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(3, [], [])

        _, kwargs = strategy._checkpoint_store.save_round_history.call_args
        assert kwargs["rounds"] == [3]
        assert kwargs["training_id"] == 70
