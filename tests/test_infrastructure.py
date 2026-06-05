"""
test_infrastructure.py
Testes unitários e de integração para a camada de infraestrutura.

Cobre:
  - schedule_state.py               : SchedulerState (persistência, serialização)
  - client_availability_checker.py  : ClientAvailabilityChecker (registry, ping)
  - round_training_fl_dispatcher.py : RoundDispatcher (dispatch, polling, convergência)
  - scheduler_daemon.py             : FederatedScheduler (job_round, quórum, parada)
  - heartbeat.py                    : write_heartbeat (criação, atualização, timestamp)
  - infrastructure/server/strategy.py : ConvergenceTracker (produção), ProductionFedProxStrategy

Uso:
    pytest tests/test_infrastructure.py -v
    pytest tests/test_infrastructure.py -v -k "TestSchedulerState"
"""
import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── adiciona raiz do projeto e src/ ao sys.path ──────────────────────────────
# infrastructure/ não tem __init__.py (namespace package Python 3),
# mas suas subpastas têm. O ROOT precisa estar no sys.path para que
# 'from infrastructure.mosaicfl_scheduler.* import ...' funcione.
# NÃO adicionar os subdiretórios diretamente: scheduler_daemon.py usa
# relative imports que falham fora do contexto de pacote (chamaria sys.exit(1)).
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── v2 ──────────────────────────────────────────────────────────────────────
from mosaicfl.v2.config import (
    BATCH_SIZE, CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
    DEVICE, EMBED_DIM, LR, MAX_SEQ_LEN, NUM_CLASSES,
    NUM_HEADS, NUM_LAYERS, PROXIMAL_MU, VOCAB_SIZE,
)
from mosaicfl.v2.model_v2 import BEHRTEncoderLayer, PositionalEncoding, SimplifiedBEHRT
from mosaicfl.v2.server_v2 import ConvergenceTracker, get_evaluate_fn, start_server, weighted_average
from mosaicfl.v2.client_v2 import FedProxClient, create_client_fn

# ── infrastructure ───────────────────────────────────────────────────────────
from infrastructure.mosaicfl_scheduler.schedule_state import SchedulerState, DEFAULT_STATE_PATH
from infrastructure.mosaicfl_scheduler.client_availability_checker import ClientAvailabilityChecker
from infrastructure.mosaicfl_scheduler.round_training_fl_dispatcher import (
    RoundDispatcher, CONVERGENCE_THRESHOLD, CONVERGENCE_PATIENCE,
)
from infrastructure.mosaicfl_scheduler.scheduler_daemon import FederatedScheduler
import infrastructure.mosaicfl_client.heartbeat as heartbeat_mod


# ═══════════════════════════════════════════════════════════════════════════════
# SchedulerState
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchedulerState:

    def test_default_values(self):
        s = SchedulerState()
        assert s.total_rounds_completed == 0
        assert s.current_round == 0
        assert s.converged is False
        assert s.convergence_round is None
        assert s.accuracy_history == []
        assert s.last_run is None
        assert s.client_history == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        s = SchedulerState(
            total_rounds_completed=7,
            accuracy_history=[0.70, 0.75, 0.78],
            converged=False,
            last_run="2026-06-04T02:00:00",
        )
        s.save(path)
        loaded = SchedulerState.load(path)
        assert loaded.total_rounds_completed == 7
        assert loaded.accuracy_history == [0.70, 0.75, 0.78]
        assert loaded.last_run == "2026-06-04T02:00:00"
        assert loaded.converged is False

    def test_load_nonexistent_file_returns_default(self, tmp_path):
        path = tmp_path / "nao_existe.json"
        s = SchedulerState.load(path)
        assert s.total_rounds_completed == 0
        assert s.converged is False

    def test_save_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "state.json"
        s = SchedulerState(total_rounds_completed=1)
        s.save(nested)
        assert nested.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        path = tmp_path / "state.json"
        s = SchedulerState(total_rounds_completed=3, converged=True, convergence_round=3)
        s.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_rounds_completed"] == 3
        assert data["converged"] is True
        assert data["convergence_round"] == 3

    def test_convergence_persists_across_reload(self, tmp_path):
        path = tmp_path / "state.json"
        s = SchedulerState()
        s.converged = True
        s.convergence_round = 8
        s.save(path)
        reloaded = SchedulerState.load(path)
        assert reloaded.converged is True
        assert reloaded.convergence_round == 8

    def test_accuracy_history_persists(self, tmp_path):
        path = tmp_path / "state.json"
        accs = [0.60, 0.65, 0.70, 0.72]
        s = SchedulerState(accuracy_history=accs)
        s.save(path)
        reloaded = SchedulerState.load(path)
        assert reloaded.accuracy_history == accs

    def test_multiple_save_load_cycles(self, tmp_path):
        path = tmp_path / "state.json"
        s = SchedulerState()
        for i in range(5):
            s.total_rounds_completed = i + 1
            s.accuracy_history.append(0.7 + i * 0.01)
            s.save(path)
            s = SchedulerState.load(path)
        assert s.total_rounds_completed == 5
        assert len(s.accuracy_history) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# ClientAvailabilityChecker
# ═══════════════════════════════════════════════════════════════════════════════

class TestClientAvailabilityChecker:

    def test_no_registry_returns_zero(self, tmp_path):
        checker = ClientAvailabilityChecker()
        count, active = checker.check_via_server(str(tmp_path / "nao_existe.json"))
        assert count == 0
        assert active == []

    def test_recent_clients_counted(self, tmp_path):
        registry = {
            "hosp_a": {"last_seen": time.time() - 60, "status": "ready"},
            "hosp_b": {"last_seen": time.time() - 120, "status": "ready"},
        }
        f = tmp_path / "registry.json"
        f.write_text(json.dumps(registry))
        checker = ClientAvailabilityChecker()
        count, active = checker.check_via_server(str(f))
        assert count == 2
        assert "hosp_a" in active and "hosp_b" in active

    def test_stale_clients_excluded(self, tmp_path):
        registry = {
            "hosp_recente": {"last_seen": time.time() - 60, "status": "ready"},
            "hosp_antigo":  {"last_seen": time.time() - 700, "status": "offline"},
        }
        f = tmp_path / "registry.json"
        f.write_text(json.dumps(registry))
        checker = ClientAvailabilityChecker()
        count, active = checker.check_via_server(str(f))
        assert count == 1
        assert "hosp_recente" in active
        assert "hosp_antigo" not in active

    def test_all_stale_returns_zero(self, tmp_path):
        registry = {
            "h1": {"last_seen": time.time() - 700},
            "h2": {"last_seen": time.time() - 900},
        }
        f = tmp_path / "registry.json"
        f.write_text(json.dumps(registry))
        checker = ClientAvailabilityChecker()
        count, _ = checker.check_via_server(str(f))
        assert count == 0

    def test_corrupted_json_returns_zero(self, tmp_path):
        f = tmp_path / "registry.json"
        f.write_text("{ invalid json {{")
        checker = ClientAvailabilityChecker()
        count, active = checker.check_via_server(str(f))
        assert count == 0
        assert active == []

    def test_ping_offline_host_returns_zero(self):
        checker = ClientAvailabilityChecker(known_clients=["127.0.0.1:19999"])
        count, active = checker.check_via_ping(timeout=0.1)
        assert count == 0
        assert active == []

    def test_register_client_no_duplicate(self):
        checker = ClientAvailabilityChecker()
        checker.register_client("h1", "192.168.1.10", 8081)
        checker.register_client("h1", "192.168.1.10", 8081)
        assert checker.known_clients.count("192.168.1.10:8081") == 1

    def test_register_multiple_distinct_clients(self):
        checker = ClientAvailabilityChecker()
        checker.register_client("h1", "192.168.1.10", 8081)
        checker.register_client("h2", "192.168.1.11", 8081)
        assert len(checker.known_clients) == 2

    def test_empty_registry_file(self, tmp_path):
        f = tmp_path / "registry.json"
        f.write_text("{}")
        checker = ClientAvailabilityChecker()
        count, active = checker.check_via_server(str(f))
        assert count == 0
        assert active == []


# ═══════════════════════════════════════════════════════════════════════════════
# RoundDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoundDispatcher:
    """
    CORREÇÃO: O original tentava patchar
        'infrastructure.mosaicfl_scheduler.round_training_fl_dispatcher.SchedulerState'
    mas SchedulerState é importado LOCALMENTE dentro de __init__ (não no topo do módulo),
    então esse atributo não existe no namespace do módulo.

    A solução é patchar no módulo de origem:
        'infrastructure.mosaicfl_scheduler.schedule_state.SchedulerState'
    """

    def _make_dispatcher(self, state=None):
        s = state or SchedulerState()
        with patch("infrastructure.mosaicfl_scheduler.schedule_state.SchedulerState") as MockState:
            MockState.load.return_value = s
            d = RoundDispatcher(server_address="localhost:8080")
        d.state = s  # sobrescreve estado no caso de mock imperfeito
        return d, s

    def test_check_convergence_insufficient_history(self):
        d, s = self._make_dispatcher()
        s.accuracy_history = [0.70, 0.71]
        assert not d.check_convergence()

    def test_check_convergence_stable_accuracies(self):
        d, s = self._make_dispatcher()
        s.accuracy_history = [0.800, 0.8001, 0.8002, 0.8000]
        s.total_rounds_completed = 4
        assert d.check_convergence() is True
        assert s.converged is True
        assert s.convergence_round == 4

    def test_check_convergence_unstable(self):
        d, s = self._make_dispatcher()
        s.accuracy_history = [0.60, 0.70, 0.65, 0.80]
        assert not d.check_convergence()
        assert s.converged is False

    def test_convergence_round_not_overwritten(self):
        d, s = self._make_dispatcher()
        s.accuracy_history = [0.800, 0.8001, 0.8002, 0.8000]
        s.total_rounds_completed = 4
        s.converged = True
        s.convergence_round = 4
        d.check_convergence()
        assert s.convergence_round == 4

    def test_dispatch_round_returns_true_on_metrics(self):
        d, s = self._make_dispatcher()
        metrics = {"round": 1, "accuracy": 0.75, "loss": 0.42}
        d._poll_round_metrics = MagicMock(return_value=metrics)
        result = d.dispatch_round(1, ["h0", "h1", "h2"])
        assert result is True
        assert s.accuracy_history == [0.75]
        assert s.total_rounds_completed == 1

    def test_dispatch_round_returns_false_on_no_metrics(self):
        d, s = self._make_dispatcher()
        d._poll_round_metrics = MagicMock(return_value=None)
        assert d.dispatch_round(1, ["h0"]) is False

    def test_dispatch_round_saves_state_on_success(self):
        d, s = self._make_dispatcher()
        metrics = {"round": 2, "accuracy": 0.80, "loss": 0.30}
        d._poll_round_metrics = MagicMock(return_value=metrics)
        s.save = MagicMock()
        d.dispatch_round(2, ["h0", "h1"])
        s.save.assert_called_once()

    def test_dispatch_round_does_not_save_on_failure(self):
        d, s = self._make_dispatcher()
        d._poll_round_metrics = MagicMock(return_value=None)
        s.save = MagicMock()
        d.dispatch_round(1, ["h0"])
        s.save.assert_not_called()

    def test_poll_round_metrics_returns_callable(self):
        """_poll_round_metrics deve existir e ser callable."""
        d, _ = self._make_dispatcher()
        assert callable(d._poll_round_metrics)

    def test_check_convergence_needs_patience_plus_one_values(self):
        d, s = self._make_dispatcher()
        s.accuracy_history = [0.8] * CONVERGENCE_PATIENCE
        assert not d.check_convergence()
        s.accuracy_history = [0.8] * (CONVERGENCE_PATIENCE + 1)
        assert d.check_convergence()


# ═══════════════════════════════════════════════════════════════════════════════
# FederatedScheduler
# ═══════════════════════════════════════════════════════════════════════════════

class TestFederatedScheduler:
    """
    CORREÇÃO principal: FederatedScheduler._job_round chama _check_server_connectivity()
    que tenta uma conexão TCP real com self.server_address.
    O __init__ original não definia self.server_address → AttributeError.
    Após corrigir o código de produção (server_address adicionado ao __init__),
    ainda precisamos mockar _check_server_connectivity nos testes para evitar
    tentativas de conexão de rede.
    """

    def _make_scheduler(
        self,
        num_available=3,
        dispatch_success=True,
        converge_after_dispatch=False,
        state=None,
    ):
        checker = MagicMock()
        checker.check_via_server.return_value = (
            num_available,
            [f"h{i}" for i in range(num_available)],
        )
        dispatcher = MagicMock()
        dispatcher.dispatch_round.return_value = dispatch_success
        dispatcher.check_convergence.return_value = converge_after_dispatch
        s = state or SchedulerState()

        with patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerState") as MockState, \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.ClientAvailabilityChecker",
                   return_value=checker), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.RoundDispatcher",
                   return_value=dispatcher):
            MockState.load.return_value = s
            scheduler = FederatedScheduler(interval_hours=1, min_clients=3, max_rounds=20)
            scheduler.state = s

        # Mock de conectividade — sem conexão TCP real nos testes
        scheduler._check_server_connectivity = MagicMock(return_value=True)
        return scheduler, checker, dispatcher, s

    def test_job_round_skips_when_converged(self):
        s = SchedulerState(); s.converged = True
        sched, checker, dispatcher, _ = self._make_scheduler(state=s)
        sched._stop_scheduler = MagicMock()
        sched._job_round()
        checker.check_via_server.assert_not_called()
        dispatcher.dispatch_round.assert_not_called()
        sched._stop_scheduler.assert_called_once()

    def test_job_round_skips_when_max_rounds_reached(self):
        s = SchedulerState(); s.total_rounds_completed = 20
        sched, checker, dispatcher, _ = self._make_scheduler(state=s)
        sched._stop_scheduler = MagicMock()
        sched._job_round()
        checker.check_via_server.assert_not_called()
        dispatcher.dispatch_round.assert_not_called()
        sched._stop_scheduler.assert_called_once()

    def test_job_round_skips_when_insufficient_clients(self):
        sched, checker, dispatcher, _ = self._make_scheduler(num_available=2)
        sched._job_round()
        dispatcher.dispatch_round.assert_not_called()

    def test_job_round_dispatches_when_quorum_met(self):
        sched, checker, dispatcher, s = self._make_scheduler(num_available=3)
        sched._job_round()
        dispatcher.dispatch_round.assert_called_once_with(1, ["h0", "h1", "h2"])

    def test_job_round_checks_convergence_after_dispatch(self):
        sched, _, dispatcher, _ = self._make_scheduler(num_available=3)
        sched._job_round()
        dispatcher.check_convergence.assert_called_once()

    def test_job_round_stops_on_convergence(self):
        sched, _, dispatcher, _ = self._make_scheduler(
            num_available=3, converge_after_dispatch=True
        )
        sched._stop_scheduler = MagicMock()
        sched._job_round()
        sched._stop_scheduler.assert_called_once()

    def test_job_round_increments_to_next_round(self):
        s = SchedulerState(); s.total_rounds_completed = 3
        sched, _, dispatcher, _ = self._make_scheduler(state=s)
        sched._job_round()
        args = dispatcher.dispatch_round.call_args[0]
        assert args[0] == 4  # 3 + 1

    def test_stop_scheduler_sets_should_stop_flag(self):
        sched, *_ = self._make_scheduler()
        sched.scheduler = None
        sched._stop_scheduler()
        assert sched._should_stop is True

    def test_stop_scheduler_pauses_apscheduler_job(self):
        sched, *_ = self._make_scheduler()
        mock_ap = MagicMock()
        sched.scheduler = mock_ap
        sched._stop_scheduler()
        mock_ap.pause_job.assert_called_once_with("federated_round")

    def test_stop_scheduler_handles_no_apscheduler(self):
        sched, *_ = self._make_scheduler()
        sched.scheduler = None
        sched._stop_scheduler()
        assert sched._should_stop is True

    def test_heartbeat_is_callable(self):
        sched, *_ = self._make_scheduler()
        assert callable(sched._heartbeat)

    def test_run_once_calls_job_round(self):
        sched, *_ = self._make_scheduler()
        sched._job_round = MagicMock()
        sched.run_once()
        sched._job_round.assert_called_once()

    def test_job_round_does_not_check_convergence_on_dispatch_failure(self):
        """Se dispatch falhar, convergência não deve ser verificada."""
        sched, _, dispatcher, _ = self._make_scheduler(
            num_available=3, dispatch_success=False
        )
        sched._job_round()
        dispatcher.check_convergence.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Heartbeat
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeartbeat:

    def test_creates_registry_file(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_test")
        heartbeat_mod.write_heartbeat(status="ready", registry_path=registry_path)
        assert Path(registry_path).exists()

    def test_writes_correct_status(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_a")
        heartbeat_mod.write_heartbeat(status="training", registry_path=registry_path)
        with open(registry_path) as f:
            data = json.load(f)
        assert data["hosp_a"]["status"] == "training"

    def test_timestamp_is_recent(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_ts")
        before = time.time()
        heartbeat_mod.write_heartbeat(status="ready", registry_path=registry_path)
        after = time.time()
        with open(registry_path) as f:
            data = json.load(f)
        ts = data["hosp_ts"]["last_seen"]
        assert before <= ts <= after

    def test_preserves_other_clients(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        existing = {"outro_hosp": {"last_seen": 1000.0, "status": "ready", "client_id": "outro"}}
        Path(registry_path).write_text(json.dumps(existing))
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "novo_hosp")
        heartbeat_mod.write_heartbeat(status="ready", registry_path=registry_path)
        with open(registry_path) as f:
            data = json.load(f)
        assert "outro_hosp" in data
        assert "novo_hosp" in data

    def test_handles_corrupted_json(self, tmp_path, monkeypatch):
        """
        CORREÇÃO: o código original em heartbeat.py escrevia JSON inválido no recovery:
            registry_file.write_text(f'{{"{CLIENT_ID}": {{(')
        Isso gerava JSONDecodeError na leitura do arquivo resultante.

        Após a correção, o recovery agora escreve um dict válido com json.dumps().
        """
        registry_path = str(tmp_path / "corrupted.json")
        Path(registry_path).write_text("{ invalid }")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_recover")
        heartbeat_mod.write_heartbeat(status="ready", registry_path=registry_path)
        with open(registry_path) as f:
            data = json.load(f)
        assert "hosp_recover" in data
        assert data["hosp_recover"]["status"] == "ready"

    def test_multiple_status_updates(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_update")
        for status in ["ready", "training", "done"]:
            heartbeat_mod.write_heartbeat(status=status, registry_path=registry_path)
        with open(registry_path) as f:
            data = json.load(f)
        assert data["hosp_update"]["status"] == "done"

    def test_client_id_written_to_registry(self, tmp_path, monkeypatch):
        registry_path = str(tmp_path / "registry.json")
        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_id_test")
        heartbeat_mod.write_heartbeat(status="ready", registry_path=registry_path)
        with open(registry_path) as f:
            data = json.load(f)
        assert data["hosp_id_test"]["client_id"] == "hosp_id_test"


# ═══════════════════════════════════════════════════════════════════════════════
# ProductionFedProxStrategy (infrastructure/server/strategy.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProductionConvergenceTracker:

    @pytest.fixture
    def tracker(self):
        from infrastructure.mosaicfl_server.strategy import ConvergenceTracker as ProdTracker
        return ProdTracker(threshold=0.005, patience=3)

    def test_not_enough_history(self, tracker):
        assert not tracker.check(0.70)
        assert not tracker.check(0.71)
        assert not tracker.check(0.72)

    def test_converges_with_stable_accuracy(self, tracker):
        for acc in [0.80, 0.8001, 0.8002, 0.8000]:
            result = tracker.check(acc)
        assert result is True
        assert tracker.converged_round is not None

    def test_does_not_converge_with_large_delta(self, tracker):
        for acc in [0.60, 0.70, 0.65, 0.80]:
            result = tracker.check(acc)
        assert result is False

    def test_reset_clears_state(self, tracker):
        tracker.check(0.80); tracker.check(0.80)
        tracker.check(0.80); tracker.check(0.80)
        tracker.reset()
        assert tracker.history == []
        assert tracker.converged_round is None


class TestProductionFedProxStrategy:
    """
    CORREÇÃO: o original tentava patchar
        'infrastructure.mosaicfl_server.strategy.FedProx.__init__'
    mas FedProx não é importado como atributo do módulo — está acessível via
    'fl.server.strategy.FedProx'. O alvo correto do patch é:
        'flwr.server.strategy.FedProx.__init__'

    Também foram corrigidos:
      - 'import strategy as strat_mod' → 'import infrastructure.mosaicfl_server.strategy as strat_mod'
      - 'from infrastructure.mosaicfl_server.strategy import strategy as strat_mod' (syntax inválida)
      - O uso de patch("...LOG_DIR") e patch("...CHECKPOINT_DIR") para variáveis de módulo
    """

    @pytest.fixture
    def strategy_and_model(self, tmp_path):
        from infrastructure.mosaicfl_server.strategy import (
            ProductionFedProxStrategy, ConvergenceTracker,
        )

        model = SimplifiedBEHRT(use_cls_token=True)
        with patch("flwr.server.strategy.FedProx.__init__", return_value=None):
            strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
            strategy.global_model = model
            strategy.tracker = ConvergenceTracker()
            strategy.round_counter = 0
            strategy.should_stop = False
            strategy.CHECKPOINT_DIR = tmp_path / "checkpoints"
            strategy.LOG_DIR = tmp_path / "logs"
            strategy.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            strategy.LOG_DIR.mkdir(parents=True, exist_ok=True)
        return strategy, model, tmp_path

    def test_load_global_weights_updates_model(self, strategy_and_model):
        strategy, model, _ = strategy_and_model
        zero_params = [np.zeros_like(v.cpu().numpy()) for v in model.state_dict().values()]
        strategy._load_global_weights(zero_params)
        for v in model.state_dict().values():
            if v.dtype == torch.float32:
                assert torch.allclose(v, torch.zeros_like(v))

    def test_aggregate_evaluate_writes_metrics_file(self, strategy_and_model):
        """
        CORREÇÃO: usa patch() nas variáveis de módulo LOG_DIR e CHECKPOINT_DIR
        em vez de tentar importar 'strategy as strat_mod' diretamente.
        """
        strategy, model, tmp_path = strategy_and_model
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        import infrastructure.mosaicfl_server.strategy as strat_mod
        old_log = strat_mod.LOG_DIR
        strat_mod.LOG_DIR = log_dir
        strategy.LOG_DIR = log_dir
        try:
            with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                       return_value=(0.4, {"accuracy": 0.78})):
                strategy.aggregate_evaluate(2, [], [])
        finally:
            strat_mod.LOG_DIR = old_log

        metrics_file = log_dir / "round_2_metrics.json"
        assert metrics_file.exists()
        with open(metrics_file) as f:
            data = json.load(f)
        assert data["round"] == 2
        assert data["accuracy"] == 0.78

    def test_aggregate_evaluate_sets_should_stop_on_convergence(self, strategy_and_model):
        strategy, _, tmp_path = strategy_and_model
        import infrastructure.mosaicfl_server.strategy as strat_mod
        old_log = strat_mod.LOG_DIR
        strat_mod.LOG_DIR = tmp_path / "logs"
        strategy.LOG_DIR = tmp_path / "logs"
        # Força convergência prévia no tracker
        strategy.tracker.history = [0.80] * (strategy.tracker.patience + 1)
        strategy.tracker.converged_round = 5
        try:
            with patch("flwr.server.strategy.FedProx.aggregate_evaluate",
                       return_value=(0.3, {"accuracy": 0.80})):
                strategy.aggregate_evaluate(6, [], [])
        finally:
            strat_mod.LOG_DIR = old_log
        assert strategy.should_stop is True

    def test_aggregate_fit_saves_checkpoint(self, strategy_and_model):
        strategy, model, tmp_path = strategy_and_model
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        import infrastructure.mosaicfl_server.strategy as strat_mod
        old_ckpt = strat_mod.CHECKPOINT_DIR
        strat_mod.CHECKPOINT_DIR = checkpoint_dir
        strategy.CHECKPOINT_DIR = checkpoint_dir
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        try:
            with patch("flwr.server.strategy.FedProx.aggregate_fit",
                       return_value=(params, {})):
                strategy.aggregate_fit(3, [], [])
        finally:
            strat_mod.CHECKPOINT_DIR = old_ckpt

        checkpoint = checkpoint_dir / "round_3.pt"
        assert checkpoint.exists()

    def test_load_weights_strict_false_no_crash(self, strategy_and_model):
        """strict=False deve carregar sem RuntimeError mesmo com chaves faltando."""
        strategy, model, _ = strategy_and_model
        all_values = list(model.state_dict().values())
        partial_params = [np.zeros_like(v.cpu().numpy()) for v in all_values[:3]]
        try:
            strategy._load_global_weights(partial_params)
        except RuntimeError:
            pytest.fail("_load_global_weights lançou RuntimeError com strict=False")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRAÇÃO: Scheduler + State + Availability + Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchedulerIntegration:

    def _scheduler_with_connectivity_mock(self, state, checker, dispatcher):
        """Helper: cria FederatedScheduler com mocks injetados e TCP desabilitado."""
        with patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerState") as MockState, \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.ClientAvailabilityChecker",
                   return_value=checker), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.RoundDispatcher",
                   return_value=dispatcher):
            MockState.load.return_value = state
            scheduler = FederatedScheduler(interval_hours=1, min_clients=3, max_rounds=20)
            scheduler.state = state
        scheduler._check_server_connectivity = MagicMock(return_value=True)
        return scheduler

    def test_state_survives_scheduler_restart(self, tmp_path):
        path = tmp_path / "state.json"
        s1 = SchedulerState()
        s1.total_rounds_completed = 5
        s1.accuracy_history = [0.70, 0.72, 0.74, 0.75, 0.76]
        s1.save(path)
        s2 = SchedulerState.load(path)
        assert s2.total_rounds_completed == 5
        assert s2.accuracy_history == [0.70, 0.72, 0.74, 0.75, 0.76]

    def test_converged_state_prevents_new_rounds(self, tmp_path):
        s = SchedulerState()
        s.converged = True; s.convergence_round = 8; s.total_rounds_completed = 8

        checker = MagicMock()
        dispatcher = MagicMock()
        scheduler = self._scheduler_with_connectivity_mock(s, checker, dispatcher)
        scheduler._stop_scheduler = MagicMock()

        scheduler._job_round()
        checker.check_via_server.assert_not_called()
        dispatcher.dispatch_round.assert_not_called()
        scheduler._stop_scheduler.assert_called_once()

    def test_client_goes_offline_then_online(self, tmp_path, monkeypatch):
        """
        CORREÇÃO: o original usava 'import heartbeat as hb' que falha pois
        o módulo está em infrastructure.mosaicfl_client.heartbeat.
        Usa heartbeat_mod (já importado no topo do arquivo).
        """
        registry_file = tmp_path / "registry.json"
        checker = ClientAvailabilityChecker()

        monkeypatch.setattr(heartbeat_mod, "CLIENT_ID", "hosp_volatile")
        heartbeat_mod.write_heartbeat(status="ready", registry_path=str(registry_file))
        count_before, _ = checker.check_via_server(str(registry_file))
        assert count_before == 1

        # Simula cliente estale
        with open(registry_file) as f:
            data = json.load(f)
        data["hosp_volatile"]["last_seen"] = time.time() - 700
        with open(registry_file, "w") as f:
            json.dump(data, f)
        count_after, _ = checker.check_via_server(str(registry_file))
        assert count_after == 0

        # Cliente volta online
        heartbeat_mod.write_heartbeat(status="ready", registry_path=str(registry_file))
        count_back, _ = checker.check_via_server(str(registry_file))
        assert count_back == 1

    def test_full_round_cycle_updates_state(self, tmp_path):
        state = SchedulerState()
        metrics = {"round": 1, "accuracy": 0.75, "loss": 0.42}

        checker = MagicMock()
        checker.check_via_server.return_value = (3, ["h0", "h1", "h2"])

        def mock_dispatch(round_num, clients):
            state.accuracy_history.append(metrics["accuracy"])
            state.total_rounds_completed = round_num
            return True

        dispatcher = MagicMock()
        dispatcher.dispatch_round.side_effect = mock_dispatch
        dispatcher.check_convergence.return_value = False

        scheduler = self._scheduler_with_connectivity_mock(state, checker, dispatcher)
        scheduler._job_round()

        assert state.total_rounds_completed == 1
        assert state.accuracy_history == [0.75]
        dispatcher.dispatch_round.assert_called_once_with(1, ["h0", "h1", "h2"])
        dispatcher.check_convergence.assert_called_once()

    def test_sequential_rounds_increment_correctly(self, tmp_path):
        """
        CORREÇÃO: o original tinha 'scheduler_daemon.scheduler_daemon.RoundDispatcher'
        (módulo duplicado), que causava AttributeError. O correto é
        'scheduler_daemon.RoundDispatcher' (sem repetição).
        """
        state = SchedulerState()
        called_rounds = []

        def mock_dispatch(round_num, clients):
            called_rounds.append(round_num)
            state.accuracy_history.append(0.70 + round_num * 0.01)
            state.total_rounds_completed = round_num
            return True

        checker = MagicMock()
        checker.check_via_server.return_value = (3, ["h0", "h1", "h2"])
        dispatcher = MagicMock()
        dispatcher.dispatch_round.side_effect = mock_dispatch
        dispatcher.check_convergence.return_value = False

        scheduler = self._scheduler_with_connectivity_mock(state, checker, dispatcher)

        for _ in range(3):
            scheduler._job_round()

        assert called_rounds == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
