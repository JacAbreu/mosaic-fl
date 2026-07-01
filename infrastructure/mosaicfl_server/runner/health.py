"""health.py — Publicação de status de saúde do servidor (endpoint /healthz + arquivo local)."""
import json
import logging
from datetime import datetime

from .config import LOG_DIR, SERVER_ADDRESS, _health

logger = logging.getLogger(__name__)


def write_health_status(status: str, round_num: int = 0, clients: int = 0):
    """Atualiza o endpoint /healthz e persiste status em arquivo."""
    state = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
        "connected_clients": clients,
        "address": SERVER_ADDRESS,
    }
    _health.set_status(**state)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_DIR / "server_health.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.debug("Erro health check: %s", e)
