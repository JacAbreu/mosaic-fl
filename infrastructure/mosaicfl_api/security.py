"""Segurança: autenticação JWT/API-Key, pseudonimização de patient_id, rate limiting."""
import hashlib
import hmac
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from . import audit

logger = logging.getLogger(__name__)

# ── Configuração ─────────────────────────────────────────────────────────────
_AUTH_REQUIRED  = os.getenv("FL_AUTH_REQUIRED", "true").lower() not in ("false", "0", "no")
_PID_SECRET     = os.getenv("FL_PATIENT_ID_SECRET", "")
_JWT_SECRET     = os.getenv("FL_JWT_SECRET", "")
_JWT_PUBLIC_KEY = ""
_jwt_key_file   = os.getenv("FL_JWT_PUBLIC_KEY_FILE", "")
_JWT_AUDIENCE   = os.getenv("FL_JWT_AUDIENCE", "mosaicfl")
_JWT_ISSUER     = os.getenv("FL_JWT_ISSUER", "")

if _jwt_key_file and Path(_jwt_key_file).exists():
    _JWT_PUBLIC_KEY = Path(_jwt_key_file).read_text(encoding="utf-8")

_JWT_LIB = None
try:
    import jwt as _jwt_lib_import
    _JWT_LIB = _jwt_lib_import
except ImportError:
    pass

# ── Pseudonimização (LGPD Art. 13 §4º) ───────────────────────────────────────

def _pid_to_internal(raw_patient_id: str) -> str:
    """Converte patient_id recebido para ID interno via HMAC-SHA256."""
    if not _PID_SECRET:
        return raw_patient_id
    return hmac.new(_PID_SECRET.encode(), raw_patient_id.encode(), hashlib.sha256).hexdigest()


# ── JWT ───────────────────────────────────────────────────────────────────────

def _validate_jwt(token: str) -> None:
    """Valida JWT se FL_JWT_SECRET ou FL_JWT_PUBLIC_KEY_FILE estiver configurado."""
    if not _JWT_LIB or not (_JWT_SECRET or _JWT_PUBLIC_KEY):
        return
    try:
        key        = _JWT_PUBLIC_KEY or _JWT_SECRET
        algorithms = ["RS256", "RS512"] if _JWT_PUBLIC_KEY else ["HS256"]
        options: dict = {}
        if _JWT_ISSUER:
            options["issuer"] = _JWT_ISSUER
        _JWT_LIB.decode(
            token, key, algorithms=algorithms,
            audience=_JWT_AUDIENCE, options=options,
        )
    except Exception as exc:
        logger.warning("jwt_validation_failed: %s", exc)
        raise HTTPException(status_code=403, detail="Token inválido ou expirado")


# ── Rate limiting (janela deslizante por IP, sem dependências externas) ───────

class _SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self._max    = max_calls
        self._window = window_seconds
        self._log: dict[str, list] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now    = time.monotonic()
        calls  = self._log[key]
        cutoff = now - self._window
        while calls and calls[0] < cutoff:
            calls.pop(0)
        if len(calls) >= self._max:
            return False
        calls.append(now)
        return True


_api_limiter    = _SlidingWindowLimiter(max_calls=120, window_seconds=60.0)
_ingest_limiter = _SlidingWindowLimiter(max_calls=30,  window_seconds=60.0)


async def _rate_check(request, limiter: _SlidingWindowLimiter) -> None:
    ip = request.client.host if request.client else "unknown"
    if not limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Limite de requisições excedido. Tente novamente em breve.",
        )


# ── FastAPI auth dependency ───────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_header  = HTTPBearer(auto_error=False)


async def _get_token_fingerprint(
    api_key: Optional[str] = Security(_api_key_header),
    bearer:  Optional[HTTPAuthorizationCredentials] = Security(_bearer_header),
) -> str:
    """Extrai token, valida JWT (se configurado) e retorna fingerprint SHA-256."""
    token = api_key or (bearer.credentials if bearer else None)
    if _AUTH_REQUIRED and not token:
        raise HTTPException(status_code=403, detail="Token de autorização ausente")
    if not token:
        return "dev-mode"
    _validate_jwt(token)
    return audit.token_fingerprint(token)
