"""config.py — Constantes de ambiente, singleton do HealthServer e configuração de logging do cliente."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from infrastructure.shared.health_server import HealthServer

SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "localhost:8080")
CLIENT_ID = os.getenv("FL_CLIENT_ID", "client_0")
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
HEALTH_PORT = int(os.getenv("FL_HEALTH_PORT", "8081"))

HEARTBEAT_INTERVAL = int(os.getenv("FL_HEARTBEAT_INTERVAL", "60"))
RECONNECT_DELAY = int(os.getenv("FL_RECONNECT_DELAY", "30"))

logger = logging.getLogger(__name__)

_health = HealthServer(port=HEALTH_PORT)


def setup_logging(client_id: str) -> None:
    """Configura logging em arquivo rotativo e stdout (idempotente se já configurado)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if logging.getLogger().handlers:
        return
    file_handler = RotatingFileHandler(
        LOG_DIR / f"client_{client_id}.log",
        maxBytes=20 * 1024 * 1024,  # 20 MB por arquivo
        backupCount=5,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler(sys.stdout)],
    )
