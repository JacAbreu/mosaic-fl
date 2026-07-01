"""config.py — Configuração via variáveis de ambiente e setup de logging do scheduler."""
import logging
import os
import sys
from pathlib import Path

LOG_FILE = Path(os.getenv("FL_SCHEDULER_LOG", "logs/scheduler.log"))

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# Configurações via variáveis de ambiente
SCHEDULER_INTERVAL_HOURS = float(os.getenv("FL_SCHEDULER_INTERVAL_HOURS", "6"))
SCHEDULER_TIMEZONE = os.getenv("FL_SCHEDULER_TIMEZONE", "America/Sao_Paulo")
MIN_AVAILABLE_CLIENTS = int(os.getenv("FL_SCHEDULER_MIN_CLIENTS", "3"))
MAX_ROUNDS = int(os.getenv("FL_SCHEDULER_MAX_ROUNDS", "20"))
CONVERGENCE_THRESHOLD = float(os.getenv("FL_SCHEDULER_CONV_THRESHOLD", "0.005"))
CONVERGENCE_PATIENCE = int(os.getenv("FL_SCHEDULER_CONV_PATIENCE", "3"))
