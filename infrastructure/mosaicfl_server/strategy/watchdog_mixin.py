"""watchdog_mixin.py — Recovery de estado (TrainingState) e watchdog de timeout por round.

_save_state fica em core.py (não aqui) porque lê CHECKPOINT_DIR — os testes fazem
patch direto em "infrastructure.mosaicfl_server.strategy.core.CHECKPOINT_DIR", então
o método que lê essa constante precisa estar no mesmo módulo onde ela é definida.
"""
import logging
import threading

logger = logging.getLogger(__name__)


class _WatchdogMixin:
    """Requer os atributos definidos em ProductionFedProxStrategy.__init__
    (tracker, round_counter, _current_state, _round_timeout, _round_timer) e o
    método _save_state (de core.py, via herança na classe final)."""

    def _restore_from_state(self, state) -> None:
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
