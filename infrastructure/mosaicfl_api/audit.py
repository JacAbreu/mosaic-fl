"""
audit.py — Registro de auditoria para conformidade LGPD Art. 37.

Emite um registro estruturado por cada acesso a dado de paciente:
  operação, patient_id pseudonimizado, fingerprint do token, timestamp ISO-8601.

Pseudonimização + fingerprint permitem que o IAM hospitalar correlacione
identidade↔acesso sem que este módulo armazene PII em texto claro.
O log de auditoria é gravado em arquivo separado do log de aplicação
e nunca é propagado para handlers de nível superior.
"""
import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
_AUDIT_FILE = _LOG_DIR / "audit.log"

_audit = logging.getLogger("mosaicfl.audit")


def _setup() -> None:
    """Configura handler de arquivo (idempotente)."""
    if _audit.handlers:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        _AUDIT_FILE, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    try:
        try:
            from pythonjsonlogger.json import JsonFormatter
        except ImportError:
            from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[no-redef]

        fmt = JsonFormatter(
            "%(asctime)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    except ImportError:
        fmt = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",%(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    handler.setFormatter(fmt)
    _audit.addHandler(handler)
    _audit.setLevel(logging.INFO)
    _audit.propagate = False  # nunca mistura com logs de aplicação


def pseudonymize(patient_id: str) -> str:
    """SHA-256 truncado do patient_id — identificador opaco, não reversível por este módulo."""
    return hashlib.sha256(patient_id.encode()).hexdigest()[:16]


def token_fingerprint(token: str) -> str:
    """SHA-256 truncado do token — permite correlação pelo IAM sem expor a credencial."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def log_access(
    operation: str,
    token_fp: str,
    patient_id: str | None = None,
    **kwargs,
) -> None:
    """
    Emite registro de auditoria LGPD Art. 37.

    Args:
        operation:  "predict" | "ingest" | "patient_read" | "patient_list" | "model_reload"
        token_fp:   fingerprint do token de autorização (12 hex chars)
        patient_id: patient_id real — pseudonimizado antes de gravar
        **kwargs:   campos extras (ex: exam_count=5, risk_score=0.72)
    """
    _setup()
    extra: dict = {
        "event": "patient_data_access",
        "operation": operation,
        "token_fingerprint": token_fp,
    }
    if patient_id is not None:
        extra["patient_id_hash"] = pseudonymize(patient_id)
    extra.update(kwargs)
    _audit.info("audit", extra=extra)
