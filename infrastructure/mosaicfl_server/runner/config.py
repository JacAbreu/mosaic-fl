"""config.py — Constantes de ambiente, singleton do HealthServer e configuração de logging."""
import os
from pathlib import Path

from infrastructure.shared.health_server import HealthServer
from infrastructure.shared.logging_setup import setup_logging as _setup_logging

SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
HEALTH_PORT = int(os.getenv("FL_HEALTH_PORT", "8081"))

_health = HealthServer(port=HEALTH_PORT)


def setup_logging() -> None:
    """Configura logging estruturado (JSON ou texto) via logging_setup central."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_file="server_daemon.log")
