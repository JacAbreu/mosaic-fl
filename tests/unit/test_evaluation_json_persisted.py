"""
Testes para o achado #6 da auditoria Caminho A vs B (2026-07-28):
fl_checkpoints.evaluation_json sempre ficava NULL no Caminho B — Caminho A monta
um payload rico (manual_loop.py:494-510), Caminho B nunca passava evaluation_json
pro checkpoint_store.save(). Corrigido via ProductionFedProxStrategy._build_
evaluation_json(), chamado nos dois pontos que salvam checkpoint (aggregate_
evaluate() no melhor round, e _persist_federated_calibration() no fim do treino).
"""
import sys
from pathlib import Path
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
    strategy._ece_pre = 0.08
    strategy._ece = 0.11
    strategy._best_f1_macro = 0.0
    strategy._best_macro_auc = None
    strategy._best_accuracy = 0.0
    strategy._best_criterion_value = 0.0
    strategy._best_round = 0
    strategy._best_state_dict = None
    strategy._best_confusion_matrix = None
    strategy._rdp_accountant = None
    strategy._dp_epsilon_simple = None
    strategy._dp_last_group_multipliers = None
    strategy.CHECKPOINT_DIR = tmp_path / "checkpoints"
    strategy.LOG_DIR = tmp_path / "logs"
    strategy.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strategy.LOG_DIR.mkdir(parents=True, exist_ok=True)
    from infrastructure.mosaicfl_server.state_store import TrainingState
    strategy._current_state = TrainingState()
    return strategy


class TestBuildEvaluationJson:
    def test_includes_core_fields(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig())

        strategy = _make_strategy(tmp_path)
        strategy.round_counter = 42
        strategy._best_f1_macro = 0.45
        strategy._best_accuracy = 0.78
        strategy._best_macro_auc = 0.83

        payload = strategy._build_evaluation_json(best_round=30, best_per_class_f1=[0.8, 0.0, 0.0, 0.6, 0.5])

        assert payload["best_round"] == 30
        assert payload["total_rounds_so_far"] == 42
        assert payload["best_f1_macro"] == 0.45
        assert payload["best_accuracy"] == 0.78
        assert payload["best_macro_auc"] == 0.83
        assert payload["best_per_class_f1"] == [0.8, 0.0, 0.0, 0.6, 0.5]
        assert payload["ece"] == 0.11
        assert payload["ece_pre"] == 0.08
        assert len(payload["class_labels"]) == 5

    def test_confusion_matrix_stats_none_when_never_set(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig())

        strategy = _make_strategy(tmp_path)
        payload = strategy._build_evaluation_json(best_round=1, best_per_class_f1=None)
        assert payload["confusion_matrix_stats"] is None

    def test_confusion_matrix_stats_derived_when_present(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig())

        strategy = _make_strategy(tmp_path)
        strategy._best_confusion_matrix = [
            [10, 0, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 6, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 2],
        ]
        payload = strategy._build_evaluation_json(best_round=1, best_per_class_f1=None)
        stats = payload["confusion_matrix_stats"]
        assert stats is not None
        assert stats["accuracy_p_hat"] == 1.0
        assert stats["n_total"] == 30
        assert "accuracy_ci_95_wilson" in stats

    def test_dp_fields_none_when_dp_disabled(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=0.0))

        strategy = _make_strategy(tmp_path)
        payload = strategy._build_evaluation_json(best_round=1, best_per_class_f1=None)

        assert payload["dp_noise_multiplier"] is None
        assert payload["dp_epsilon_simple"] is None
        assert payload["dp_noise_strategy"] is None

    def test_dp_epsilon_rdp_computed_when_accountant_present(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=0.5))

        strategy = _make_strategy(tmp_path)
        strategy._rdp_accountant = MagicMock()
        strategy._rdp_accountant.get_epsilon.return_value = 12.3

        payload = strategy._build_evaluation_json(best_round=1, best_per_class_f1=None)

        assert payload["dp_epsilon_rdp"] == 12.3

    def test_accountant_error_does_not_raise(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=0.5))

        strategy = _make_strategy(tmp_path)
        strategy._rdp_accountant = MagicMock()
        strategy._rdp_accountant.get_epsilon.side_effect = RuntimeError("boom")

        payload = strategy._build_evaluation_json(best_round=1, best_per_class_f1=None)  # não deve levantar

        assert payload["dp_epsilon_rdp"] is None


class TestEvaluationJsonReachesCheckpointSave:
    def test_best_round_save_includes_evaluation_json(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig())

        strategy = _make_strategy(tmp_path)
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.7, "f1_macro": 0.5})):
            strategy.aggregate_evaluate(25, [], [])

        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["evaluation_json"] is not None
        assert kwargs["evaluation_json"]["best_round"] == 25

    def test_calibration_save_includes_evaluation_json(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig())

        strategy = _make_strategy(tmp_path)
        strategy._best_round = 30
        strategy._best_accuracy = 0.8

        strategy._persist_federated_calibration(
            server_round=110,
            calibration_method="temperature",
            aggregated_metrics={"calibration_method": "temperature", "temperature": 1.5},
        )

        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["evaluation_json"] is not None
        assert kwargs["evaluation_json"]["best_round"] == 30
