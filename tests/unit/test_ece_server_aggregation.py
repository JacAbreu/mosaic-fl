"""
Testes para a agregação federada de ece/ece_pre em
ProductionFedProxStrategy.aggregate_evaluate() — achado 2026-07-26: ece/ece_pre
ficavam sempre None/0 em produção porque o único cálculo existente exigia um
test_loader centralizado (indisponível por design de privacidade no Caminho B).
Cada cliente manda só contagens agregadas por bin (client.py::
local_ece_bin_stats); aqui só somamos entre hospitais, nunca centralizamos
predição/rótulo bruto.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.evaluation import local_ece_bin_stats
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(tmp_path):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    from mosaicfl.core.model import SimplifiedBEHRT
    from mosaicfl.core.convergence import ConvergenceTracker

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
    strategy._num_rounds = 1
    strategy._training_completed = False
    strategy._ece_pre = None
    strategy._ece = None
    strategy._best_f1_macro = 0.0
    strategy._best_macro_auc = None
    strategy._best_accuracy = 0.0
    strategy._best_criterion_value = 0.0
    strategy._best_round = 0
    strategy._train_start_time = 0.0
    strategy.CHECKPOINT_DIR = tmp_path / "checkpoints"
    strategy.LOG_DIR = tmp_path / "logs"
    strategy.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strategy.LOG_DIR.mkdir(parents=True, exist_ok=True)
    from infrastructure.mosaicfl_server.state_store import TrainingState
    strategy._current_state = TrainingState()
    return strategy


def _evaluate_result(client_id, ece_pre_bins=None, ece_post_bins=None, **extra_metrics):
    metrics = {"accuracy": 0.7, "client_id": client_id, **extra_metrics}
    if ece_pre_bins is not None:
        metrics["ece_pre_bin_stats_json"] = json.dumps(ece_pre_bins)
    if ece_post_bins is not None:
        metrics["ece_post_bin_stats_json"] = json.dumps(ece_post_bins)
    return (None, SimpleNamespace(metrics=metrics))


class TestEceServerAggregation:
    def test_no_ece_metrics_leaves_ece_none(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        results = [_evaluate_result(0), _evaluate_result(1)]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])
        assert strategy._ece_pre is None
        assert strategy._ece is None

    def test_aggregates_ece_pre_across_two_clients(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        bins_a = local_ece_bin_stats(
            __import__("torch").full((10,), 0.9), __import__("torch").ones(10).bool()
        )
        bins_b = local_ece_bin_stats(
            __import__("torch").full((10,), 0.9), __import__("torch").zeros(10).bool()
        )
        results = [
            _evaluate_result(0, ece_pre_bins=bins_a),
            _evaluate_result(1, ece_pre_bins=bins_b),
        ]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])

        # 10 corretos (conf 0.9) + 10 errados (conf 0.9) = acc 0.5, conf_media 0.9, gap 0.4
        assert strategy._ece_pre == 0.4

    def test_aggregates_ece_post_across_two_clients(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        import torch
        bins_a = local_ece_bin_stats(torch.full((10,), 1.0), torch.ones(10).bool())
        bins_b = local_ece_bin_stats(torch.full((10,), 1.0), torch.ones(10).bool())
        results = [
            _evaluate_result(0, ece_post_bins=bins_a),
            _evaluate_result(1, ece_post_bins=bins_b),
        ]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])
        assert strategy._ece == 0.0

    def test_only_one_client_reports_ece_still_aggregates(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        import torch
        bins_a = local_ece_bin_stats(torch.full((5,), 1.0), torch.ones(5).bool())
        results = [_evaluate_result(0, ece_pre_bins=bins_a), _evaluate_result(1)]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            strategy.aggregate_evaluate(1, results, [])
        assert strategy._ece_pre == 0.0

    def test_malformed_json_does_not_crash_aggregate_evaluate(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        results = [(None, SimpleNamespace(metrics={
            "accuracy": 0.7, "client_id": 0, "ece_pre_bin_stats_json": "{not valid json",
        }))]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})):
            try:
                strategy.aggregate_evaluate(1, results, [])
            except Exception as e:
                assert False, f"não deveria propagar exceção: {e}"

    def test_training_completion_passes_ece_to_update_evaluation_metrics(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        import torch
        bins = local_ece_bin_stats(torch.full((10,), 0.8), torch.ones(10).bool())
        results = [_evaluate_result(0, ece_pre_bins=bins, ece_post_bins=bins)]
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7})), \
             patch.object(strategy, "_save_federated_training_id_marker"):
            strategy.aggregate_evaluate(1, results, [])

        _, kwargs = strategy._checkpoint_store.update_evaluation_metrics.call_args
        assert kwargs["ece_pre"] == 0.2
        assert kwargs["ece"] == 0.2
