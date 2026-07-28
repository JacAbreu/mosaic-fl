"""
Testes para o achado 2026-07-28 (auditoria Caminho A vs B): a convergência em
ProductionFedProxStrategy sempre checava accuracy, ignorando
FED_CFG.checkpoint_criterion — Caminho A usa f1_global quando o critério é
"f1_macro" (manual_loop.py:278, default do projeto). Não muda quantas rodadas o
Flower executa (loop fixo), mas converged/convergence_round gravados no banco
usavam um critério diferente do que best_round/best_accuracy já usam.
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
    strategy._ece_pre = None
    strategy._ece = None
    strategy._best_f1_macro = 0.0
    strategy._best_macro_auc = None
    strategy._best_accuracy = 0.0
    strategy._best_criterion_value = 0.0
    strategy._best_round = 0
    strategy._best_state_dict = None
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


class TestConvergenceUsesCheckpointCriterion:
    def test_f1_macro_criterion_feeds_tracker_with_f1_macro(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(checkpoint_criterion="f1_macro"))

        strategy = _make_strategy(tmp_path)
        strategy.tracker.check = MagicMock(wraps=strategy.tracker.check)

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.9, "f1_macro": 0.3})):
            strategy.aggregate_evaluate(25, [], [])

        strategy.tracker.check.assert_called_once_with(0.3)  # f1_macro, não accuracy

    def test_accuracy_criterion_feeds_tracker_with_accuracy(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(checkpoint_criterion="accuracy"))

        strategy = _make_strategy(tmp_path)
        strategy.tracker.check = MagicMock(wraps=strategy.tracker.check)

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.9, "f1_macro": 0.3})):
            strategy.aggregate_evaluate(25, [], [])

        strategy.tracker.check.assert_called_once_with(0.9)  # accuracy, não f1_macro

    def test_convergence_criterion_matches_best_round_selection_criterion(self, tmp_path, monkeypatch):
        """Acoplamento direto com o achado: convergência e seleção de melhor
        rodada (best_criterion_value) devem usar o MESMO critério — antes da
        correção, podiam divergir (convergência sempre accuracy, seleção
        respeitava checkpoint_criterion)."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(checkpoint_criterion="f1_macro"))

        strategy = _make_strategy(tmp_path)
        # accuracy alta mas f1_macro baixo — se convergência usasse accuracy,
        # divergiria da seleção de melhor rodada (que já usa f1_macro).
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": 0.95, "f1_macro": 0.2})):
            strategy.aggregate_evaluate(25, [], [])

        assert strategy._best_criterion_value == 0.2  # f1_macro
        assert strategy.tracker.history[-1] == 0.2     # convergência viu o mesmo valor


class TestMinRoundsWarmup:
    """Achado 2026-07-28: Caminho A suspende avaliação de convergência até
    FED_CFG.min_rounds (manual_loop.py:300-301) — Caminho B nunca teve esse gate,
    então ConvergenceTracker (patience=3 default) podia convergir já na rodada 4."""

    def test_convergence_never_checked_before_min_rounds(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(min_rounds=20))

        strategy = _make_strategy(tmp_path)
        strategy.tracker.check = MagicMock(wraps=strategy.tracker.check)

        # Métrica idêntica em todas as rodadas — convergeria quase imediatamente
        # se o warm-up não estivesse ativo (delta=0 < threshold, patience=3).
        for round_num in range(1, 15):
            with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                       return_value=(0.4, {"accuracy": 0.7, "f1_macro": 0.7})):
                strategy.aggregate_evaluate(round_num, [], [])

        strategy.tracker.check.assert_not_called()
        assert strategy.tracker.converged_round is None

    def test_convergence_evaluated_after_min_rounds(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(min_rounds=5))

        strategy = _make_strategy(tmp_path)

        for round_num in range(1, 12):
            with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                       return_value=(0.4, {"accuracy": 0.7, "f1_macro": 0.7})):
                strategy.aggregate_evaluate(round_num, [], [])

        # Rodadas 1-5: warm-up, história vazia. Rodadas 6-11: 6 valores idênticos
        # -> converge (patience=3, threshold=0.005, delta=0 em todos).
        assert len(strategy.tracker.history) == 6  # só rodadas 6..11
        assert strategy.tracker.converged_round is not None
