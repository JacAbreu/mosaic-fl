"""
heartbeat.py
Registra status do cliente para o scheduler monitorar.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
CLIENT_ID = os.getenv("FL_CLIENT_ID", "client_0")


def write_heartbeat(status: str = "ready", registry_path: Optional[str] = None):
    """Escreve status no registry compartilhado."""
    registry_file = Path(registry_path) if registry_path else LOG_DIR / "client_registry.json"
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    registry = {}
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            pass

    registry[CLIENT_ID] = {
        "last_seen": datetime.now().timestamp(),
        "status": status,
        "client_id": CLIENT_ID,
    }

    try:
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.debug(f"Erro ao escrever heartbeat: {e}")