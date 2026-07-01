"""
scheduler_daemon — Scheduler de rounds federados usando APScheduler.

Padrão de execução:
  1. Desperta (trigger: intervalo ou horário específico)
  2. Verifica clientes disponíveis
  3. Se >= min_available_clients: dispara round
  4. Se < min_available_clients: loga e volta a dormir
  5. Verifica convergência
  6. Se convergiu ou max_rounds: para agendamento

Uso:
    python scheduler_daemon.py              # modo daemon (loop infinito)
    python scheduler_daemon.py --once       # modo cron (1 execução e termina)

Submódulos:
  config.py           — variáveis de ambiente e setup de logging
  core.py                — FederatedScheduler (__init__, _check_server_connectivity, _job_round)
  lifecycle_mixin.py        — start_daemon, run_once, _heartbeat, _stop_scheduler
  cli.py                       — main()
"""
from .config import (
    CONVERGENCE_PATIENCE,
    CONVERGENCE_THRESHOLD,
    LOG_FILE,
    MAX_ROUNDS,
    MIN_AVAILABLE_CLIENTS,
    SCHEDULER_INTERVAL_HOURS,
    SCHEDULER_TIMEZONE,
)
from .core import ClientAvailabilityChecker, FederatedScheduler, RoundDispatcher, SchedulerState, SchedulerStateStore
from .cli import main

__all__ = [
    "FederatedScheduler",
    "main",
    "SchedulerState",
    "SchedulerStateStore",
    "ClientAvailabilityChecker",
    "RoundDispatcher",
    "LOG_FILE",
    "SCHEDULER_INTERVAL_HOURS",
    "SCHEDULER_TIMEZONE",
    "MIN_AVAILABLE_CLIENTS",
    "MAX_ROUNDS",
    "CONVERGENCE_THRESHOLD",
    "CONVERGENCE_PATIENCE",
]
