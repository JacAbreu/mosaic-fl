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
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── adiciona raiz do projeto e src/ ao sys.path ──────────────────────────────
# infrastructure/ não tem __init__.py (namespace package Python 3),
# mas suas subpastas têm. O ROOT precisa estar no sys.path para que
# 'from infrastructure.mosaicfl_scheduler.* import ...' funcione.
# NÃO adicionar os subdiretórios diretamente: scheduler_daemon.py usa
# relative imports que falham fora do contexto de pacote (chamaria sys.exit(1)).
ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── v2 ──────────────────────────────────────────────────────────────────────
from mosaicfl.core.config import (
    BATCH_SIZE, CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
    DEVICE, EMBED_DIM, LR, MAX_SEQ_LEN, NUM_CLASSES,
    NUM_HEADS, NUM_LAYERS, PROXIMAL_MU, VOCAB_SIZE,
)
from mosaicfl.core.model import BEHRTEncoderLayer, PositionalEncoding, SimplifiedBEHRT
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.federated import get_evaluate_fn, weighted_average
from experiments.experiment_server import start_server
from mosaicfl.core.client import FedProxClient, create_client_fn
from mosaicfl.core.preprocessor import EHRPreprocessor, split_by_institution

# ── infrastructure ───────────────────────────────────────────────────────────
from infrastructure.mosaicfl_scheduler.schedule_state import SchedulerState, DEFAULT_STATE_PATH
from infrastructure.mosaicfl_scheduler.client_availability_checker import ClientAvailabilityChecker
from infrastructure.mosaicfl_scheduler.round_training_fl_dispatcher import RoundDispatcher
from infrastructure.mosaicfl_scheduler.scheduler_daemon import FederatedScheduler
import infrastructure.mosaicfl_client.heartbeat as heartbeat_mod


# ─────────────────────────────────────────────────────────────
# FIXTURES COMPARTILHADAS
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instituicao":   ["HospA", "HospA", "HospB", "HospB", "HospC"],
        "idade":         [25.0, 6.0, 45.0, 365.0, 70.0],
        "idade_unidade": ["anos", "meses", "anos", "dias", "anos"],
        "peso":          [150.0, 70.0, 180.0, 50.0, 80.0],
        "peso_unidade":  ["lb", "kg", "lb", "kg", "kg"],
        "sintoma":       ["febre", "tosse", "dispneia", "fadiga", "mialgia"],
        "exame":         ["rt_pcr_positivo", "tomografia_normal", "rx_consolidacao",
                          "pcr_negativo", "tomografia_vidro_fosco"],
        "diagnostico":   ["covid19_leve", "covid19_moderado", "pneumonia_bacteriana",
                          "covid19_grave", "alta"],
        "desfecho":      [0, 0, 1, 1, 0],
    })


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

    def _make_dispatcher(self):
        return RoundDispatcher(server_address="localhost:8080")

    def test_check_convergence_insufficient_history(self):
        d = self._make_dispatcher()
        assert not d.check_convergence([0.70, 0.71])

    def test_check_convergence_stable_accuracies(self):
        d = self._make_dispatcher()
        assert d.check_convergence([0.800, 0.8001, 0.8002, 0.8000]) is True

    def test_check_convergence_unstable(self):
        d = self._make_dispatcher()
        assert not d.check_convergence([0.60, 0.70, 0.65, 0.80])

    def test_convergence_round_not_overwritten(self):
        """check_convergence é puro — não tem efeito colateral no estado externo."""
        d = self._make_dispatcher()
        history = [0.800, 0.8001, 0.8002, 0.8000]
        assert d.check_convergence(history) is True
        # chamar novamente com o mesmo histórico ainda retorna True
        assert d.check_convergence(history) is True

    def test_dispatch_round_returns_accuracy_on_metrics(self):
        """dispatch_round retorna o float de accuracy quando métricas disponíveis."""
        d = self._make_dispatcher()
        metrics = {"round": 1, "accuracy": 0.75, "loss": 0.42}
        d._poll_round_metrics = MagicMock(return_value=metrics)
        result = d.dispatch_round(1, ["h0", "h1", "h2"])
        assert result == 0.75

    def test_dispatch_round_returns_none_on_no_metrics(self):
        """dispatch_round retorna None quando métricas não chegam."""
        d = self._make_dispatcher()
        d._poll_round_metrics = MagicMock(return_value=None)
        assert d.dispatch_round(1, ["h0"]) is None

    def test_dispatch_round_returns_none_when_accuracy_missing(self):
        """dispatch_round retorna None quando accuracy não está nas métricas."""
        d = self._make_dispatcher()
        d._poll_round_metrics = MagicMock(return_value={"round": 1, "loss": 0.5})
        assert d.dispatch_round(1, ["h0"]) is None

    def test_poll_round_metrics_returns_callable(self):
        """_poll_round_metrics deve existir e ser callable."""
        d = self._make_dispatcher()
        assert callable(d._poll_round_metrics)

    def test_check_convergence_needs_patience_plus_one_values(self):
        d = self._make_dispatcher()
        assert not d.check_convergence([0.8] * CONVERGENCE_PATIENCE)
        assert d.check_convergence([0.8] * (CONVERGENCE_PATIENCE + 1))


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
        dispatcher.dispatch_round.return_value = 0.75 if dispatch_success else None
        dispatcher.check_convergence.return_value = converge_after_dispatch
        s = state or SchedulerState()

        with patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerState") as MockState, \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerStateStore"), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.ClientAvailabilityChecker",
                   return_value=checker), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.RoundDispatcher",
                   return_value=dispatcher):
            MockState.load.return_value = s
            scheduler = FederatedScheduler(interval_hours=1, min_clients=3, max_rounds=20)
            scheduler.state = s

        scheduler._check_server_connectivity = MagicMock(return_value=True)
        scheduler._store = MagicMock()
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
            strategy.tracker = ConvergenceTracker(
                threshold=CONVERGENCE_THRESHOLD,
                patience=CONVERGENCE_PATIENCE,
            )
            strategy.round_counter = 0
            strategy.should_stop = False
            strategy.on_round_complete = None
            strategy.on_round_start = None
            strategy._state_store = None
            strategy._round_timeout = 0
            strategy._round_timer = None
            strategy._current_state = __import__(
                "infrastructure.mosaicfl_server.state_store", fromlist=["TrainingState"]
            ).TrainingState()
            strategy._last_round_metrics = {}
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
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerStateStore"), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.ClientAvailabilityChecker",
                   return_value=checker), \
             patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.RoundDispatcher",
                   return_value=dispatcher):
            MockState.load.return_value = state
            scheduler = FederatedScheduler(interval_hours=1, min_clients=3, max_rounds=20)
            scheduler.state = state
        scheduler._check_server_connectivity = MagicMock(return_value=True)
        scheduler._store = MagicMock()
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
            return metrics["accuracy"]

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
            return 0.70 + round_num * 0.01

        checker = MagicMock()
        checker.check_via_server.return_value = (3, ["h0", "h1", "h2"])
        dispatcher = MagicMock()
        dispatcher.dispatch_round.side_effect = mock_dispatch
        dispatcher.check_convergence.return_value = False

        scheduler = self._scheduler_with_connectivity_mock(state, checker, dispatcher)

        for _ in range(3):
            scheduler._job_round()

        assert called_rounds == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# ConfigLoader + Strategy — integração entre componentes
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigLoaderWithStrategy:
    """
    Verifica que configure_fit da strategy aplica corretamente o config
    retornado por um FileConfigLoader real (sem mock de ChromaDB).
    """
    from unittest.mock import MagicMock, patch

    @pytest.fixture
    def strategy_with_file_loader(self, tmp_path):
        from infrastructure.mosaicfl_server.config_loader import FileConfigLoader
        from infrastructure.mosaicfl_server.strategy import ProductionFedProxStrategy
        from infrastructure.mosaicfl_server.strategy import ConvergenceTracker as ProdTracker
        from mosaicfl.core.model import SimplifiedBEHRT
        from unittest.mock import patch, MagicMock

        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        model = SimplifiedBEHRT(use_cls_token=True)

        with patch("flwr.server.strategy.FedProx.__init__", return_value=None):
            strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
            strategy.global_model = model
            strategy.config_loader = loader
            strategy.on_round_start = None
            strategy.on_round_complete = None
            strategy.proximal_mu = 0.01
            strategy.should_stop = False
            strategy.tracker = ProdTracker(
                threshold=CONVERGENCE_THRESHOLD,
                patience=CONVERGENCE_PATIENCE,
            )
            strategy.round_counter = 0
            strategy._state_store = None
            strategy._round_timeout = 0
            strategy._round_timer = None
            strategy._current_state = __import__(
                "infrastructure.mosaicfl_server.state_store", fromlist=["TrainingState"]
            ).TrainingState()
            strategy._last_round_metrics = {}
            (tmp_path / "checkpoints").mkdir()
            (tmp_path / "logs").mkdir()

        import infrastructure.mosaicfl_server.strategy as strat_mod
        strat_mod.CHECKPOINT_DIR = tmp_path / "checkpoints"
        strat_mod.LOG_DIR = tmp_path / "logs"

        return strategy, loader

    def test_configure_fit_returns_empty_when_stop_true(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        from unittest.mock import MagicMock
        loader.write({"stop": True})
        result = strategy.configure_fit(1, MagicMock(), MagicMock())
        assert result == []
        assert strategy.should_stop is True

    def test_configure_fit_updates_proximal_mu(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        from unittest.mock import patch, MagicMock
        import pytest
        loader.write({"proximal_mu": 0.05, "stop": False})
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(1, MagicMock(), MagicMock())
        assert strategy.proximal_mu == pytest.approx(0.05)

    def test_configure_fit_no_config_delegates_to_super(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        from unittest.mock import patch, MagicMock
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]) as mock_super:
            strategy.configure_fit(1, MagicMock(), MagicMock())
        mock_super.assert_called_once()

    def test_configure_fit_calls_on_round_start_callback(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        from unittest.mock import patch, MagicMock
        import pytest
        callback = MagicMock()
        strategy.on_round_start = callback
        loader.write({"proximal_mu": 0.01, "stop": False})
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(3, MagicMock(), MagicMock())
        callback.assert_called_once_with(3, {"proximal_mu": pytest.approx(0.01), "stop": False})

    def test_configure_fit_callback_exception_does_not_propagate(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        from unittest.mock import patch, MagicMock, Mock
        strategy.on_round_start = Mock(side_effect=RuntimeError("callback falhou"))
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(1, MagicMock(), MagicMock())


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline v2 — integração multi-componente
# ═══════════════════════════════════════════════════════════════════════════════

class TestV2PipelineIntegration:
    """Pipeline completo sem RAG (sem LLM), cobrindo preprocess → model → FL."""

    def test_preprocess_to_model_forward(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        encoded_cols = [c for c in df_proc.columns if c.endswith("_encoded")]
        x = torch.tensor(df_proc[encoded_cols].values[:, :16], dtype=torch.long)
        if x.shape[1] < 16:
            pad = torch.zeros(x.shape[0], 16 - x.shape[1], dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
        model = SimplifiedBEHRT(use_cls_token=True)
        logits = model(x)
        assert logits.shape == (len(sample_df), NUM_CLASSES)

    def test_client_server_parameter_compatibility(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        server_model = SimplifiedBEHRT(use_cls_token=True)
        params_dict = zip(server_model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        missing, unexpected = server_model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0

    def test_fedavg_aggregation_preserves_shape(self):
        model = SimplifiedBEHRT(use_cls_token=True)
        sd = model.state_dict()
        states = [
            OrderedDict({k: torch.randn_like(v.float()) for k, v in sd.items()})
            for _ in range(3)
        ]
        aggregated = OrderedDict()
        for key in sd.keys():
            if sd[key].dtype in (torch.long, torch.int):
                aggregated[key] = states[0][key].to(sd[key].dtype).clamp(0, VOCAB_SIZE - 1)
            else:
                aggregated[key] = torch.stack([s[key] for s in states]).mean(0).to(sd[key].dtype)
        model.load_state_dict(aggregated, strict=True)
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        assert model(x).shape == (2, NUM_CLASSES)

    def test_convergence_tracker_in_evaluate_loop(self):
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        accuracies = [0.70, 0.72, 0.750, 0.752, 0.751, 0.750]
        converged_at = None
        for i, acc in enumerate(accuracies):
            if tracker.check(acc):
                converged_at = i + 1
                break
        assert converged_at is not None
        assert converged_at >= 4

    def test_split_then_client_then_evaluate(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma"])
        clients = split_by_institution(df_proc, num_clients=5)
        subset = clients[0]
        encoded_cols = [c for c in subset.columns if c.endswith("_encoded")]
        if not encoded_cols:
            pytest.skip("Sem colunas encoded para este subset")
        x = torch.tensor(subset[encoded_cols].values[:, :16], dtype=torch.long)
        if x.shape[1] < 16:
            pad = torch.zeros(x.shape[0], 16 - x.shape[1], dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
        y = torch.tensor(subset["desfecho"].values, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        loss, n, metrics = client.evaluate(params, {})
        assert isinstance(loss, float)
        assert n > 0
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
