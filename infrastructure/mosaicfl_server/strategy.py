"""
strategy.py
Estratégia FedProx de produção (mosaicfl.v2) com checkpoint e convergência.
"""
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch

from mosaicfl.core.config import FED_CFG
from mosaicfl.core.federated import weighted_average_accuracy, weighted_average_loss
from .config_loader import ConfigLoader, get_config_loader
from .state_store import TrainingState, TrainingStateStore

CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))

logger = logging.getLogger(__name__)


from mosaicfl.core.convergence import ConvergenceTracker


class ProductionFedProxStrategy(fl.server.strategy.FedProx):
    """
    FedProx adaptado para produção:
      - Checkpoint do modelo global a cada rodada
      - Exporta métricas para JSON (consumidas pelo scheduler)
      - Rastreia convergência
      - Lê config de runtime do PostgreSQL (ou arquivo) antes de cada round
    """

    def __init__(
        self,
        global_model: torch.nn.Module,
        vocab: Optional[Dict[str, int]] = None,
        config_loader: Optional[ConfigLoader] = None,
        on_round_start: Optional[Callable[[int, Dict], None]] = None,
        on_round_complete: Optional[Callable[[int, Dict], None]] = None,
        state_store: Optional[TrainingStateStore] = None,
        round_timeout: int = 300,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("evaluate_metrics_aggregation_fn", weighted_average_accuracy)
        kwargs.setdefault("fit_metrics_aggregation_fn", weighted_average_loss)
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.vocab: Dict[str, int] = vocab or {}
        self.config_loader: ConfigLoader = config_loader or get_config_loader()
        self.on_round_start = on_round_start
        self.on_round_complete = on_round_complete
        self.tracker = ConvergenceTracker(
            threshold=FED_CFG.convergence_threshold,
            patience=FED_CFG.convergence_patience,
        )
        self.round_counter = 0
        self.should_stop = False

        self._state_store = state_store
        self._round_timeout = round_timeout
        self._round_timer: Optional[threading.Timer] = None
        self._current_state = TrainingState()
        self._last_round_metrics: Dict = {}

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if state_store is not None:
            self._restore_from_state(state_store.load())

    def _restore_from_state(self, state: TrainingState) -> None:
        """Restaura ConvergenceTracker e estado interno a partir do estado salvo."""
        self.tracker.history = list(state.convergence_history)
        self.tracker.converged_round = state.converged_round
        self.round_counter = state.last_round
        self._current_state = state
        logger.info(
            "strategy_state_restored",
            extra={
                "previous_status": state.status,
                "last_round": state.last_round,
                "history_length": len(state.convergence_history),
                "converged_round": state.converged_round,
                "last_checkpoint": state.last_checkpoint,
            },
        )

    def _save_state(self, server_round: int) -> None:
        """Persiste estado atual no TrainingStateStore."""
        if self._state_store is None:
            return
        self._current_state.last_round = server_round
        self._current_state.convergence_history = list(self.tracker.history)
        self._current_state.converged_round = self.tracker.converged_round
        self._current_state.last_metrics = self._last_round_metrics
        self._current_state.last_checkpoint = str(CHECKPOINT_DIR / f"round_{server_round}.pt")
        self._state_store.save(self._current_state)

    def _start_round_watchdog(self, server_round: int) -> None:
        """Inicia timer que dispara se aggregate_fit não for chamado em _round_timeout s."""
        if self._round_timeout <= 0:
            return
        if self._round_timer is not None:
            self._round_timer.cancel()

        def _on_timeout() -> None:
            logger.warning(
                "round_timeout",
                extra={"round": server_round, "timeout_seconds": self._round_timeout},
            )
            self._current_state.timed_out_rounds.append(server_round)
            self._current_state.status = "running"
            self._save_state(server_round)

        self._round_timer = threading.Timer(self._round_timeout, _on_timeout)
        self._round_timer.daemon = True
        self._round_timer.start()

    def _cancel_round_watchdog(self) -> None:
        if self._round_timer is not None:
            self._round_timer.cancel()
            self._round_timer = None

    def configure_fit(
        self, server_round: int, parameters, client_manager
    ) -> List[Tuple]:
        """
        Chamado pelo Flower antes de cada round de treino.

        Lê config dinâmica do PostgreSQL (ou fallback arquivo) e aplica antes
        de delegar a seleção de clientes ao FedProx padrão.
        """
        runtime = self.config_loader.load(server_round)

        if runtime.get("stop", False):
            logger.info("round_stopped", extra={"round": server_round, "reason": "config_stop"})
            self.should_stop = True
            return []

        if "proximal_mu" in runtime and runtime["proximal_mu"] is not None:
            new_mu = float(runtime["proximal_mu"])
            if new_mu != self.proximal_mu:
                logger.info(
                    "proximal_mu_updated",
                    extra={"round": server_round, "old_mu": self.proximal_mu, "new_mu": new_mu},
                )
                self.proximal_mu = new_mu

        pause = float(runtime.get("pause_seconds", 0) or 0)
        if pause > 0:
            logger.info("round_paused", extra={"round": server_round, "pause_seconds": pause})
            time.sleep(pause)

        if self.on_round_start is not None:
            try:
                self.on_round_start(server_round, runtime)
            except Exception as e:
                logger.warning("round_start_callback_error", extra={"round": server_round, "error": str(e)})

        self._start_round_watchdog(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def _load_global_weights(self, parameters) -> None:
        """Carrega pesos agregados no modelo global (compatível com client)."""
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.global_model.state_dict().keys(), parameters)
            }
        )
        missing, unexpected = self.global_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Checkpoint: chaves não carregadas: %s", missing)
        if unexpected:
            logger.debug("Checkpoint: chaves inesperadas: %s", unexpected)

    def aggregate_fit(self, server_round, results, failures):
        """Agrega pesos e salva checkpoint. Cancela watchdog do round."""
        self._cancel_round_watchdog()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._load_global_weights(aggregated_parameters)
            checkpoint_path = CHECKPOINT_DIR / f"round_{server_round}.pt"
            from .runner import _save_checkpoint
            _save_checkpoint(
                checkpoint_path,
                {"model_state": self.global_model.state_dict(), "vocab": self.vocab},
            )
            logger.info(
                "checkpoint_saved",
                extra={
                    "round": server_round,
                    "path": str(checkpoint_path),
                    "vocab_size": len(self.vocab),
                },
            )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas e detecta convergência."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracy = aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0
        self.tracker.check(accuracy)
        self.round_counter = server_round

        converged = self.tracker.converged_round is not None
        round_metrics = {
            "round": server_round,
            "loss": aggregated_loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "converged": converged,
            "convergence_round": self.tracker.converged_round,
        }
        self._last_round_metrics = round_metrics

        metrics_file = LOG_DIR / f"round_{server_round}_metrics.json"
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(round_metrics, f, indent=2)
        except Exception as e:
            logger.warning("metrics_write_error", extra={"round": server_round, "error": str(e)})

        # Persiste estado após cada round — permite recovery exato no próximo restart
        self._current_state.status = "completed" if converged else "running"
        self._save_state(server_round)

        if self.on_round_complete is not None:
            try:
                self.on_round_complete(server_round, round_metrics)
            except Exception as e:
                logger.warning(
                    "round_complete_callback_error",
                    extra={"round": server_round, "error": str(e)},
                )

        if converged:
            self.should_stop = True
            logger.info(
                "convergence_detected",
                extra={"round": server_round, "convergence_round": self.tracker.converged_round},
            )

        return aggregated_loss, aggregated_metrics
