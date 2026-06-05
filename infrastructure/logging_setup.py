"""
logging_setup.py
Configuração centralizada de logging estruturado em JSON para MOSAIC-FL.

Uso:
    from infrastructure.logging_setup import setup_logging
    setup_logging()  # chame uma vez no entrypoint

Variáveis de ambiente:
    FL_LOG_FORMAT=json   (padrão) — saída JSON parseável por Loki/Datadog/CloudWatch
    FL_LOG_FORMAT=text   — saída humana legível (dev/debug local)
    FL_LOG_LEVEL=INFO    (padrão) — nível mínimo de log
    FL_LOG_DIR=logs      (padrão) — diretório de arquivos de log
"""
import logging
import os
import sys
from pathlib import Path

_FL_LOG_FORMAT = os.getenv("FL_LOG_FORMAT", "json").lower()
_FL_LOG_LEVEL = os.getenv("FL_LOG_LEVEL", "INFO").upper()
_FL_LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_JSON_FIELDS = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(log_file: str | None = None) -> None:
    """Configura logging estruturado (idempotente — safe chamar mais de uma vez)."""
    root = logging.getLogger()
    if root.handlers:
        return

    level = getattr(logging, _FL_LOG_LEVEL, logging.INFO)
    root.setLevel(level)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        _FL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(_FL_LOG_DIR / log_file, encoding="utf-8")
        )

    formatter = _build_formatter()
    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)


def _build_formatter() -> logging.Formatter:
    if _FL_LOG_FORMAT == "json":
        try:
            from pythonjsonlogger import jsonlogger

            return jsonlogger.JsonFormatter(
                _JSON_FIELDS,
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        except ImportError:
            pass
    return logging.Formatter(_TEXT_FORMAT)
