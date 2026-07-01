"""cli.py — Entry point de linha de comando do scheduler.

Uso:
    python scheduler_daemon.py              # modo daemon (loop infinito)
    python scheduler_daemon.py --once       # modo cron (1 execução e termina)
"""
import argparse

from .config import MAX_ROUNDS, MIN_AVAILABLE_CLIENTS, SCHEDULER_INTERVAL_HOURS, _configure_logging
from .core import FederatedScheduler


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description="Scheduler de Rounds Federados MOSAIC-FL")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Executa um único ciclo e termina (para uso com system cron)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SCHEDULER_INTERVAL_HOURS,
        help=f"Intervalo entre ciclos em horas (default: {SCHEDULER_INTERVAL_HOURS})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=MIN_AVAILABLE_CLIENTS,
        help=f"Mínimo de clientes para iniciar round (default: {MIN_AVAILABLE_CLIENTS})",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS,
        help=f"Máximo de rounds (default: {MAX_ROUNDS})",
    )
    args = parser.parse_args()

    scheduler = FederatedScheduler(
        interval_hours=args.interval,
        min_clients=args.min_clients,
        max_rounds=args.max_rounds,
    )

    if args.once:
        scheduler.run_once()
    else:
        scheduler.start_daemon()


if __name__ == "__main__":
    main()
