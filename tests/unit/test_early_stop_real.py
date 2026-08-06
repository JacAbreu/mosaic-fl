"""
Testes para o early stop real no Caminho B (achado 2026-08-01, avaliando o treino
83 — DP uniforme): convergência detectada na rodada 36 (pico, F1 macro=0,224), mas
o loop fixo de flwr.server.server.Server.fit() ("for current_round in range(1,
num_rounds + 1)", sem break/should_stop, confirmado lendo o fonte instalado do flwr
1.32.1) seguiu rodando até a 110 configurada — entre as duas, o modelo colapsou
(F1 macro=0,025) enquanto dp_epsilon_simple cresceu de ~10 pra 1065 sem nenhum
ganho de qualidade nesse intervalo.

Cobre ProductionFedProxStrategy.is_final_round() e o agendamento de
_early_stop_target_round dentro de aggregate_evaluate() — a parte "servidor
customizado quebra o loop de verdade" é testada separadamente em
test_early_stop_server.py (EarlyStoppingServer não depende do resto da
estratégia, só de _early_stop_target_round).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mosaicfl.core.config import FedConfig
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
    strategy._early_stop_target_round = None
    strategy._convergence_round = None
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
    strategy._rdp_accountant = None
    strategy._dp_epsilon_simple = None
    strategy._dp_last_group_multipliers = None
    strategy._last_rag_patterns_json = None
    strategy._rag_precision_at_k = None
    strategy._rag_k = None
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


class TestIsFinalRound:
    def test_ceiling_alone_when_early_stop_disabled(self, tmp_path):
        strategy = _make_strategy(tmp_path, num_rounds=110)
        assert strategy.is_final_round(109) is False
        assert strategy.is_final_round(110) is True
        assert strategy.is_final_round(111) is True

    def test_early_stop_target_ignored_when_flag_disabled(self, tmp_path):
        """Mesmo com um alvo setado manualmente, sem FED_CFG.early_stop=True a
        checagem de teto continua sendo a única — flag desligada preserva o
        comportamento histórico por completo."""
        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy._early_stop_target_round = 37
        assert strategy.is_final_round(37) is False
        assert strategy.is_final_round(110) is True

    def test_early_stop_target_honored_when_flag_enabled(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy._early_stop_target_round = 37
        assert strategy.is_final_round(36) is False
        assert strategy.is_final_round(37) is True
        assert strategy.is_final_round(38) is True  # >= alvo também conta


class TestEarlyStopScheduling:
    def test_not_scheduled_when_flag_disabled(self, tmp_path, monkeypatch):
        """Comportamento histórico (default) — convergência detectada, mas
        nenhum alvo de parada é agendado. is_final_round continua só o teto."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=False))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 36

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(36, [], [])

        assert strategy._early_stop_target_round is None

    def test_scheduled_one_round_after_convergence_when_flag_enabled(self, tmp_path, monkeypatch):
        """Cenário do treino 83: convergência na 36, teto em 110 — o alvo deve
        ser 37 (uma rodada extra, pra calibrate=True poder ser enviado antes
        dela rodar), não a própria 36 nem o teto 110."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 36

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(36, [], [])

        assert strategy._early_stop_target_round == 37

    def test_not_rescheduled_on_subsequent_converged_rounds(self, tmp_path, monkeypatch):
        """tracker.converged_round não reseta — toda rodada seguinte também tem
        converged=True. O agendamento só deve acontecer uma vez, na rodada em
        que a convergência foi detectada pela primeira vez."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 36

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(36, [], [])
            assert strategy._early_stop_target_round == 37
            strategy.aggregate_evaluate(37, [], [])
            assert strategy._early_stop_target_round == 37  # não virou 38

    def test_target_capped_at_num_rounds(self, tmp_path, monkeypatch):
        """Convergência detectada exatamente na última rodada configurada —
        server_round + 1 estouraria o teto; o alvo não deve passar de
        num_rounds (é a mesma rodada em que o loop já ia parar de qualquer
        jeito)."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))

        strategy = _make_strategy(tmp_path, num_rounds=5)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 5

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(5, [], [])

        assert strategy._early_stop_target_round == 5

    def test_finalizes_at_early_stop_target_not_at_ceiling(self, tmp_path, monkeypatch):
        """Fim a fim: com early_stop ligado, complete_training deve disparar
        no round-alvo (37), não esperar o teto (110) — essa é a mudança de
        comportamento real que resolve o colapso pós-convergência do treino 83."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 36

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(36, [], [])  # agenda alvo=37, não finaliza
            strategy._checkpoint_store.complete_training.assert_not_called()
            strategy.aggregate_evaluate(37, [], [])  # bate o alvo — finaliza aqui

        strategy._checkpoint_store.complete_training.assert_called_once()
        assert strategy._checkpoint_store.complete_training.call_args.kwargs["n_rounds_done"] == 37


class TestEarlyStopWithRealConvergenceTracker:
    """Reprodução do bug real do treino 84 (2026-08-01): a primeira versão do
    agendamento comparava `self.tracker.converged_round == server_round`, mas
    ConvergenceTracker.converged_round é `len(self.history)` (convergence.py),
    NÃO o número da rodada — history só cresce a partir de min_rounds+1
    (aggregate_evaluate só chama tracker.check() quando server_round >
    FED_CFG.min_rounds). No treino 84 (min_rounds=20), a convergência real
    foi detectada na rodada 38, mas tracker.converged_round valia 18 — a
    comparação nunca batia, o alvo nunca era agendado. Os testes anteriores
    (TestEarlyStopScheduling acima) não pegaram isso porque setavam
    tracker.converged_round manualmente IGUAL a server_round, mascarando
    exatamente essa discrepância. Esta classe roda tracker.check() de
    verdade, sem hand-set, forçando as duas quantidades a divergir como no
    treino 84."""

    def test_target_uses_real_round_number_even_when_tracker_index_diverges(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=True))  # min_rounds=20 (default)

        strategy = _make_strategy(tmp_path, num_rounds=110)
        assert FedConfig().min_rounds == 20  # documenta a premissa deste teste

        # Rounds 1-20: warm-up, tracker.check() nem é chamado (não importa o valor).
        # Rounds 21+: accuracy estável — converge assim que patience (3) rounds
        # consecutivos tiverem delta < threshold. tracker.converged_round (índice
        # interno) vai ficar bem menor que o server_round real onde isso acontece.
        def _fake_aggregate_evaluate(self_, server_round, results, failures):
            accuracy = 0.30 if server_round <= 20 else 0.70
            return 0.3, {"accuracy": accuracy}

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate", _fake_aggregate_evaluate):
            triggering_round = None
            for r in range(1, 31):
                strategy.aggregate_evaluate(r, [], [])
                if strategy._early_stop_target_round is not None:
                    triggering_round = r
                    break

        assert triggering_round is not None, "convergência nunca foi detectada — ajustar o cenário do teste"
        # A prova de que o bug do treino 84 não se repete: tracker.converged_round
        # (índice interno na janela) e triggering_round (rodada real do FL)
        # DIVERGEM — e mesmo assim o alvo agendado é baseado na rodada real.
        assert strategy.tracker.converged_round < triggering_round
        assert strategy._early_stop_target_round == triggering_round + 1


class TestConvergenceRoundPersistence:
    """migration 035 — fl_trainings.convergence_round. Independente de
    FED_CFG.early_stop: mesmo com a flag desligada (comportamento default),
    a rodada real de convergência deve ser rastreada e persistida em
    complete_training(), pra dar pra comparar best_round × convergence_round
    em /fl-training-results (achado real no treino 85: best_round=4,
    convergence_round=74 — divergência grande, "convergência" != "boa
    qualidade" sob DP)."""

    def test_convergence_round_tracks_real_round_not_tracker_index(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=False))  # default

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 18  # índice interno, propositalmente != 38

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(38, [], [])

        assert strategy._convergence_round == 38  # rodada real, não o índice 18

    def test_convergence_round_not_reset_on_subsequent_rounds(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=False))

        strategy = _make_strategy(tmp_path, num_rounds=110)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 18

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(38, [], [])
            assert strategy._convergence_round == 38
            strategy.aggregate_evaluate(39, [], [])
            assert strategy._convergence_round == 38  # não vira 39

    def test_convergence_round_passed_to_complete_training(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=False))

        strategy = _make_strategy(tmp_path, num_rounds=5)
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 2  # índice interno, propositalmente != 5

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.80})):
            strategy.aggregate_evaluate(5, [], [])

        strategy._checkpoint_store.complete_training.assert_called_once()
        assert strategy._checkpoint_store.complete_training.call_args.kwargs["convergence_round"] == 5

    def test_convergence_round_none_when_never_converged(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(early_stop=False))

        strategy = _make_strategy(tmp_path, num_rounds=3)

        with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                   return_value=(0.3, {"accuracy": 0.60})):
            strategy.aggregate_evaluate(3, [], [])

        strategy._checkpoint_store.complete_training.assert_called_once()
        assert strategy._checkpoint_store.complete_training.call_args.kwargs["convergence_round"] is None
