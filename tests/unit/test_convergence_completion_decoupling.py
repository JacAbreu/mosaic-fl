"""
Testes para a separação entre "convergência detectada" e "treino finalizado"
em ProductionFedProxStrategy.aggregate_evaluate() — achado 2026-07-27 avaliando
o treino 74: convergência foi detectada na rodada 88, mas flwr/server/server.py
roda um loop FIXO ("for current_round in range(1, num_rounds + 1)") sem nenhum
mecanismo de saída antecipada — nem self.should_stop, nem configure_fit()/
configure_evaluate() retornando [] interrompem esse for. O treino continuou até
a rodada 110 de verdade (onde calibrate=True dispara e ece/ece_pre são calculados
pela primeira vez). O código antigo finalizava (complete_training/
update_evaluation_metrics) assim que converged=True, na rodada 88 — descartando
o ece/ece_pre real, calculado só depois. Corrigido: finaliza só em is_last_round.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(tmp_path, num_rounds):
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
    strategy._num_rounds = num_rounds
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


class TestConvergenceDoesNotFinalizeEarly:
    def test_convergence_alone_does_not_call_complete_training(self, tmp_path):
        """Rodada converge mas ainda não é a última configurada — não deve
        finalizar (mesmo padrão do treino 74: convergiu na 88, num_rounds=110)."""
        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 88

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(88, [], [])

        strategy._checkpoint_store.complete_training.assert_not_called()
        strategy._checkpoint_store.update_evaluation_metrics.assert_not_called()
        assert strategy._training_completed is False

    def test_still_marks_should_stop_and_logs_convergence(self, tmp_path, caplog):
        """should_stop continua sendo setado (informativo, alguns testes/lugares
        dependem disso) mesmo não finalizando de verdade."""
        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 88

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(88, [], [])

        assert strategy.should_stop is True

    def test_finalizes_at_true_last_round_after_earlier_convergence(self, tmp_path):
        """Depois de convergir cedo (88) mas o loop seguindo até num_rounds (110,
        já que o Flower não para sozinho), a finalização real acontece em 110 —
        com ece/ece_pre computados nessa rodada, não descartados."""
        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 88

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(88, [], [])   # convergência, não finaliza
            strategy._ece_pre = 0.05                  # simula calibração real na 110
            strategy._ece = 0.03
            strategy.aggregate_evaluate(110, [], [])  # última rodada de verdade

        strategy._checkpoint_store.complete_training.assert_called_once()
        kwargs = strategy._checkpoint_store.complete_training.call_args.kwargs
        assert kwargs["n_rounds_done"] == 110
        assert kwargs["converged"] is True  # ainda registra que convergiu, só não finalizou lá

        eval_kwargs = strategy._checkpoint_store.update_evaluation_metrics.call_args.kwargs
        assert eval_kwargs["ece_pre"] == 0.05
        assert eval_kwargs["ece"] == 0.03

    def test_finalizes_immediately_when_convergence_and_last_round_coincide(self, tmp_path):
        """Caso comum em testes/treinos curtos: convergência detectada bem na
        última rodada configurada — finaliza normalmente, sem regressão."""
        strategy = _make_strategy(tmp_path, num_rounds=5)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 5

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(5, [], [])

        strategy._checkpoint_store.complete_training.assert_called_once()
        assert strategy._checkpoint_store.complete_training.call_args.kwargs["n_rounds_done"] == 5

    def test_finalizes_at_last_round_even_without_convergence(self, tmp_path):
        """Nunca convergiu (converged=False o treino todo) — finaliza mesmo assim
        quando bate a última rodada configurada, comportamento pré-existente
        preservado (ver treino 72: converged=False, n_rounds_done=110)."""
        strategy = _make_strategy(tmp_path, num_rounds=3)

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.60})):
            strategy.aggregate_evaluate(3, [], [])

        strategy._checkpoint_store.complete_training.assert_called_once()
        assert strategy._checkpoint_store.complete_training.call_args.kwargs["converged"] is False
