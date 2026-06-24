"""Endpoints administrativos: /api/fl/status e /api/fl/reload. Startup checks."""
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from .. import audit
from .. import state
from ..inference_engine import InferenceEngine
from ..schemas import FLStatus
from ..security import _get_token_fingerprint

logger = logging.getLogger(__name__)

router = APIRouter()


async def startup_checks() -> None:
    """Valida configuração crítica — chamado no lifespan antes de aceitar tráfego."""
    from sqlalchemy import text as _text

    _env            = os.getenv("FL_ENV", "development").lower()
    _cors_origins   = os.getenv("FL_CORS_ORIGINS", "*").split(",")
    _pid_secret     = os.getenv("FL_PATIENT_ID_SECRET", "")
    _jwt_secret     = os.getenv("FL_JWT_SECRET", "")
    _jwt_pub_key    = ""
    _jwt_pub_file   = os.getenv("FL_JWT_PUBLIC_KEY_FILE", "")
    from pathlib import Path
    if _jwt_pub_file and Path(_jwt_pub_file).exists():
        _jwt_pub_key = Path(_jwt_pub_file).read_text(encoding="utf-8")

    errors: list[str] = []

    if not os.getenv("FL_DB_URL"):
        errors.append("FL_DB_URL não configurado")
    else:
        try:
            with state._db._engine.connect() as conn:
                conn.execute(_text("SELECT 1"))
        except Exception as exc:
            errors.append(f"Banco inacessível: {exc}")

    if _env == "production":
        if not _pid_secret:
            errors.append(
                "FL_PATIENT_ID_SECRET não configurado — patient_id seria armazenado em texto "
                "claro (viola LGPD Art. 13 §4º)"
            )
        if "*" in _cors_origins:
            errors.append(
                "FL_CORS_ORIGINS='*' não é permitido em FL_ENV=production — "
                "configure domínios explícitos"
            )
        if not (_jwt_secret or _jwt_pub_key):
            logger.warning(
                "startup_warning: FL_JWT_SECRET / FL_JWT_PUBLIC_KEY_FILE não configurados — "
                "autenticação valida apenas presença do token"
            )

    if errors:
        for msg in errors:
            logger.critical("startup_check_failed: %s", msg)
        raise RuntimeError(f"Configuração inválida para inicialização: {errors}")

    if not _pid_secret and _env != "production":
        logger.warning(
            "startup_warning: FL_PATIENT_ID_SECRET não configurado — "
            "patient_id será armazenado sem pseudonimização (apenas dev/testes)"
        )
    if "*" in _cors_origins and _env != "production":
        logger.warning(
            "startup_warning: FL_CORS_ORIGINS='*' — configure domínios explícitos antes de produção"
        )

    logger.warning(
        "startup_warning: rate_limiter=in_process — em produção com múltiplos workers "
        "(Gunicorn/Uvicorn), o limite é por processo e não global. "
        "Configure FL_REDIS_URL e substitua _SlidingWindowLimiter por fastapi-limiter + Redis."
    )
    logger.info(
        "startup_ok env=%s auth=%s jwt=%s pid_hash=%s cors_origins=%s",
        _env,
        os.getenv("FL_AUTH_REQUIRED", "true").lower() not in ("false", "0", "no"),
        bool(_jwt_secret or _jwt_pub_key),
        bool(_pid_secret),
        _cors_origins,
    )


@router.get("/api/fl/status", response_model=FLStatus)
async def fl_status():
    ckpt = state._latest_checkpoint()
    rounds, last_updated = 0, None
    if ckpt:
        try:
            rounds = int(ckpt.stem.replace("round_", ""))
        except ValueError:
            pass
        last_updated = datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat()
    return FLStatus(
        model_ready=ckpt is not None,
        checkpoint_path=str(ckpt) if ckpt else None,
        rounds_completed=rounds,
        last_updated=last_updated,
    )


@router.post("/api/fl/reload")
async def reload_model(fingerprint: str = Depends(_get_token_fingerprint)):
    """Força recarga do checkpoint mais recente com verificação de integridade."""
    ckpt = state._latest_checkpoint()
    if not ckpt:
        raise HTTPException(status_code=404, detail="nenhum checkpoint disponível")

    if not state._verify_checkpoint_integrity(ckpt):
        logger.error("checkpoint_integrity_failed path=%s", ckpt)
        raise HTTPException(
            status_code=500,
            detail=f"Checkpoint corrompido ou adulterado: {ckpt.name}. Retrainamento necessário.",
        )

    state._engine = InferenceEngine(
        checkpoint_path=ckpt,
        db_url=os.getenv("FL_DB_URL"),
    )
    logger.info("model_reloaded", extra={"checkpoint": str(ckpt)})
    audit.log_access("model_reload", token_fp=fingerprint, checkpoint=str(ckpt))
    return {"reloaded": True, "checkpoint": str(ckpt)}
