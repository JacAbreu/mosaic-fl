"""
Testes para o bug corrigido em 2026-07-28: checkpoint_store.save() era chamado
INCONDICIONALMENTE toda rodada em aggregate_fit(), então fl_checkpoints sempre
acabava com os pesos da ÚLTIMA rodada, nunca da MELHOR (best_round/best_accuracy
ficavam corretos como metadado em fl_trainings, mas os pesos de fato salvos não
correspondiam). Caminho A (manual_loop.py:278-284) já fazia certo — só salva
dentro do "if criterion_value > best". Este arquivo cobre a mesma correção
portada pro Caminho B (ProductionFedProxStrategy).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import flwr as fl

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
    strategy._best_criterion_value = 0.0
    strategy._best_accuracy = 0.0
    strategy._best_f1_macro = 0.0
    strategy._best_macro_auc = None
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


def _real_parameters():
    model = SimplifiedBEHRT()
    ndarrays = [v.numpy() for v in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(ndarrays)


class TestAggregateFitNoLongerSavesUnconditionally:
    def test_aggregate_fit_does_not_call_checkpoint_save(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        params = _real_parameters()
        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(params, {})):
            strategy.aggregate_fit(1, [], [])
        strategy._checkpoint_store.save.assert_not_called()


class TestAggregateEvaluateSavesOnlyOnImprovement:
    def _run_evaluate(self, strategy, server_round, accuracy):
        # FED_CFG.checkpoint_criterion default é "f1_macro" — sem essa chave no
        # aggregated_metrics, criterion_value fica sempre 0.0 e nunca melhora.
        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.4, {"accuracy": accuracy, "f1_macro": accuracy})):
            strategy.aggregate_evaluate(server_round, [], [])

    def test_first_round_always_saves(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        self._run_evaluate(strategy, 1, 0.5)
        strategy._checkpoint_store.save.assert_called_once()
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["round_num"] == 1
        assert kwargs["accuracy"] == pytest.approx(0.5)

    def test_worse_round_does_not_save(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        self._run_evaluate(strategy, 1, 0.8)
        strategy._checkpoint_store.save.reset_mock()
        self._run_evaluate(strategy, 2, 0.6)  # pior que 0.8
        strategy._checkpoint_store.save.assert_not_called()

    def test_better_round_saves_again(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        self._run_evaluate(strategy, 1, 0.5)
        strategy._checkpoint_store.save.reset_mock()
        self._run_evaluate(strategy, 2, 0.7)  # melhor que 0.5
        strategy._checkpoint_store.save.assert_called_once()
        assert strategy._checkpoint_store.save.call_args.kwargs["round_num"] == 2

    def test_final_checkpoint_reflects_best_round_not_last(self, tmp_path):
        """O cenário exato do bug real: melhora na rodada 2, piora nas rodadas
        3-5 — o checkpoint final tem que continuar sendo o da rodada 2."""
        strategy = _make_strategy(tmp_path)
        self._run_evaluate(strategy, 1, 0.5)
        self._run_evaluate(strategy, 2, 0.8)  # melhor rodada
        self._run_evaluate(strategy, 3, 0.6)
        self._run_evaluate(strategy, 4, 0.65)
        self._run_evaluate(strategy, 5, 0.7)  # nunca supera 0.8

        saved_rounds = [c.kwargs["round_num"] for c in strategy._checkpoint_store.save.call_args_list]
        assert saved_rounds == [1, 2]  # só salvou nas melhoras reais, nunca 3/4/5
        assert strategy._best_round == 2
        assert strategy._best_state_dict is not None

    def test_checkpoint_save_error_does_not_raise(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        strategy._checkpoint_store.save.side_effect = RuntimeError("disco cheio")
        self._run_evaluate(strategy, 1, 0.5)  # não deve levantar exceção


class TestPersistFederatedCalibrationUsesBestStateDict:
    def test_uses_best_state_dict_not_current_round(self, tmp_path):
        import json as _json
        strategy = _make_strategy(tmp_path)
        # Simula melhor rodada já registrada com um state_dict "marcado"
        fake_best = {"marker": "sou o melhor round"}
        strategy._best_state_dict = fake_best
        strategy._best_round = 42
        strategy._best_accuracy = 0.9

        strategy._persist_federated_calibration(
            server_round=110,  # última rodada, bem depois da melhor (42)
            calibration_method="temperature",
            aggregated_metrics={"calibration_method": "temperature", "temperature": 1.5},
        )

        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["state_dict"] is fake_best
        assert kwargs["round_num"] == 42
        assert kwargs["accuracy"] == pytest.approx(0.9)

    def test_falls_back_to_current_model_if_never_improved(self, tmp_path):
        """Caso degenerado: calibração roda mas nenhuma rodada nunca melhorou o
        critério (best_state_dict continua None) — não deve quebrar."""
        strategy = _make_strategy(tmp_path)
        assert strategy._best_state_dict is None

        strategy._persist_federated_calibration(
            server_round=5,
            calibration_method="temperature",
            aggregated_metrics={"calibration_method": "temperature", "temperature": 1.2},
        )

        strategy._checkpoint_store.save.assert_called_once()
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["round_num"] == 5
