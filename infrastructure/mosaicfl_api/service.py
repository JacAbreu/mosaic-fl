"""
service.py
FastAPI — endpoint de inferência, ingestão de exames e painel web.

Segurança:
  - Autenticação: X-API-Key ou Authorization: Bearer <token>.
    Se FL_JWT_SECRET (HS256) ou FL_JWT_PUBLIC_KEY_FILE (RS256/RS512) estiver
    configurado, o token é validado criptograficamente. Sem configuração de JWT,
    exige apenas presença do token (compatível com IAM upstream que valida por proxy).
  - Pseudonimização: se FL_PATIENT_ID_SECRET estiver configurado, o patient_id
    recebido é convertido via HMAC-SHA256 antes de qualquer persistência no banco.
    Nenhum identificador real de paciente é armazenado. (LGPD Art. 13 §4º)
  - Rate limiting: janela deslizante por IP — 120 req/min geral, 30 req/min em /ingest.
  - CORS: configurar FL_CORS_ORIGINS com domínios explícitos em produção.
    Default * é rejeitado em startup se FL_ENV=production.

Persistência: PatientDB (SQLAlchemy) — PostgreSQL em produção, SQLite em dev/testes.
Auditoria: todo acesso a dado de paciente é registrado em logs/audit.log
           com patient_id pseudonimizado e fingerprint do token. (LGPD Art. 37)

Endpoints:
  POST /api/predict             — score de risco (sem persistir estado)
  POST /api/exams/ingest        — ingere + prediz (histórico completo) + exporta ClinicalPath
  GET  /api/patients            — lista paginada de pacientes com último score
  GET  /api/patients/{id}       — histórico completo + contagem de exames
  GET  /api/fl/status           — status do modelo (checkpoint, rounds)
  POST /api/fl/reload           — recarrega checkpoint mais recente
  GET  /                        — painel web
"""
import asyncio
import hashlib
import hmac
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
import math
from pydantic import BaseModel, Field, field_validator

from . import audit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# integration/clinical-path (módulo com hífen — não é package Python)
# ---------------------------------------------------------------------------
_INTEGRATION_DIR = Path(__file__).parent.parent.parent / "integration" / "clinical-path"
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))

from exporter import ClinicalPathExporter  # noqa: E402
from models import ClinicalPhase, ExamRecord, PatientExport, RiskPrediction  # noqa: E402

from .db import PatientDB  # noqa: E402
from .inference_engine import InferenceEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
_OUTPUT_DIR     = Path(os.getenv("FL_CLINICALPATH_OUTPUT", "data/clinicalpath_output"))
_CORS_ORIGINS   = os.getenv("FL_CORS_ORIGINS", "*").split(",")
_AUTH_REQUIRED  = os.getenv("FL_AUTH_REQUIRED", "true").lower() not in ("false", "0", "no")
_ENV            = os.getenv("FL_ENV", "development").lower()

# Pseudonimização de patient_id (LGPD)
_PID_SECRET = os.getenv("FL_PATIENT_ID_SECRET", "")

# JWT (opcional — se não configurado, verifica apenas presença do token)
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


# ---------------------------------------------------------------------------
# Helpers de segurança
# ---------------------------------------------------------------------------

def _pid_to_internal(raw_patient_id: str) -> str:
    """Converte patient_id recebido para ID interno via HMAC-SHA256.

    Se FL_PATIENT_ID_SECRET não estiver configurado, retorna o raw sem modificação
    (apenas para compatibilidade de desenvolvimento — nunca aceito em FL_ENV=production).
    """
    if not _PID_SECRET:
        return raw_patient_id
    return hmac.new(_PID_SECRET.encode(), raw_patient_id.encode(), hashlib.sha256).hexdigest()


def _validate_jwt(token: str) -> None:
    """Valida JWT se FL_JWT_SECRET ou FL_JWT_PUBLIC_KEY_FILE estiver configurado.

    Raises HTTPException(403) se inválido ou expirado.
    Se nenhuma chave JWT estiver configurada, não faz nada (compatível com IAM upstream).
    """
    if not _JWT_LIB or not (_JWT_SECRET or _JWT_PUBLIC_KEY):
        return

    try:
        key = _JWT_PUBLIC_KEY or _JWT_SECRET
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


# ---------------------------------------------------------------------------
# Rate limiting — janela deslizante por IP, sem dependências externas
# ---------------------------------------------------------------------------

class _SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self._max      = max_calls
        self._window   = window_seconds
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


async def _rate_check(request: Request, limiter: _SlidingWindowLimiter) -> None:
    ip = request.client.host if request.client else "unknown"
    if not limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Limite de requisições excedido. Tente novamente em breve.")


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_db             = PatientDB()
_exporter       = ClinicalPathExporter()
_engine: Optional[InferenceEngine] = None
_patient_locks: dict[str, asyncio.Lock] = {}


def _latest_checkpoint() -> Optional[Path]:
    if not _CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(_CHECKPOINT_DIR.glob("round_*.pt"))
    return ckpts[-1] if ckpts else None


def _patient_lock(pid: str) -> asyncio.Lock:
    if pid not in _patient_locks:
        _patient_locks[pid] = asyncio.Lock()
    return _patient_locks[pid]


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(
            checkpoint_path=_latest_checkpoint(),
            db_url=os.getenv("FL_DB_URL"),
        )
    return _engine


def _verify_checkpoint_integrity(path: Path) -> bool:
    """Verifica SHA-256 do checkpoint contra arquivo .sha256 (se existir)."""
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists():
        return True  # sem hash salvo — legado, aceita com aviso no caller
    actual   = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = hash_path.read_text(encoding="utf-8").strip()
    return actual == expected


# ---------------------------------------------------------------------------
# Autenticação + fingerprinting (LGPD Art. 37)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

async def _startup_checks() -> None:
    """Valida configuração crítica — chamado pelo lifespan antes de aceitar tráfego."""
    errors: list[str] = []

    if not os.getenv("FL_DB_URL"):
        errors.append("FL_DB_URL não configurado")
    else:
        try:
            from sqlalchemy import text as _text
            with _db._engine.connect() as conn:
                conn.execute(_text("SELECT 1"))
        except Exception as exc:
            errors.append(f"Banco inacessível: {exc}")

    if _ENV == "production":
        if not _PID_SECRET:
            errors.append(
                "FL_PATIENT_ID_SECRET não configurado — patient_id seria armazenado em texto "
                "claro (viola LGPD Art. 13 §4º)"
            )
        if "*" in _CORS_ORIGINS:
            errors.append(
                "FL_CORS_ORIGINS='*' não é permitido em FL_ENV=production — "
                "configure domínios explícitos"
            )
        if not (_JWT_SECRET or _JWT_PUBLIC_KEY):
            logger.warning(
                "startup_warning: FL_JWT_SECRET / FL_JWT_PUBLIC_KEY_FILE não configurados — "
                "autenticação valida apenas presença do token"
            )

    if errors:
        for msg in errors:
            logger.critical("startup_check_failed: %s", msg)
        raise RuntimeError(f"Configuração inválida para inicialização: {errors}")

    if not _PID_SECRET and _ENV != "production":
        logger.warning(
            "startup_warning: FL_PATIENT_ID_SECRET não configurado — "
            "patient_id será armazenado sem pseudonimização (apenas dev/testes)"
        )
    if "*" in _CORS_ORIGINS and _ENV != "production":
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
        _ENV, _AUTH_REQUIRED,
        bool(_JWT_SECRET or _JWT_PUBLIC_KEY),
        bool(_PID_SECRET),
        _CORS_ORIGINS,
    )


@asynccontextmanager
async def _lifespan(app):
    await _startup_checks()
    # Auto-discovery: carrega o checkpoint mais recente antes de aceitar tráfego.
    # Sem isso, a API sobe com modelo aleatório (trained=False) após qualquer reinicialização.
    engine = _get_engine()
    if engine.checkpoint_path:
        logger.info("startup_checkpoint_loaded path=%s", engine.checkpoint_path)
    else:
        logger.warning("startup_no_checkpoint — API operacional mas modelo não treinado (trained=False)")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MOSAIC-FL API",
    description="Inferência federada e painel de monitoramento clínico",
    version="0.3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ExamInput(BaseModel):
    exam_name:    str
    date:         date
    value:        float
    phase:        str = Field("IN", description="AB | EX | IN | OBITO | P_ALTA")
    ref_low:      float = 0.0
    ref_high:     float = 0.0
    origin:       Optional[str] = None
    exam_group:   Optional[str] = None
    value_text:   Optional[str] = None
    unit:         Optional[str] = None
    attendance_id: Optional[str] = None

    @field_validator("value")
    @classmethod
    def value_must_be_finite_and_non_negative(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("value não pode ser NaN ou infinito")
        if v < 0:
            raise ValueError("value não pode ser negativo")
        return v


class PredictRequest(BaseModel):
    patient_id: str
    exams:      list[ExamInput]


class ClassProbability(BaseModel):
    value:       float
    uncertainty: float


class ModelMetadata(BaseModel):
    trained:            bool          = False
    calibrated:         bool          = False
    uncertainty_method: str           = "mc_dropout"
    mc_samples:         int           = 0
    checkpoint_round:   Optional[int] = None
    checkpoint_at:      Optional[str] = None
    model_version:      Optional[str] = None
    note:               str           = (
        "Probabilidades estimadas via MC Dropout. Modelo sem calibração pós-treinamento: "
        "os valores refletem confiança relativa entre classes, não frequência empírica calibrada. "
        "Não usar como probabilidade clínica absoluta sem avaliação profissional."
    )


class PredictResponse(BaseModel):
    patient_id:          str
    risk_score:          float
    risk_date:           date
    class_probabilities: dict[str, ClassProbability]
    predicted_class:     int
    predicted_label:     str
    model_metadata:      ModelMetadata


class IngestRequest(BaseModel):
    patient_id: str
    sex:        str   = "M"
    age:        float = 0.0
    exams:      list[ExamInput]
    output_dir: Optional[str] = None


class IngestResponse(BaseModel):
    patient_id:          str
    risk_score:          float
    export_path:         str
    class_probabilities: dict[str, ClassProbability]
    predicted_class:     int
    predicted_label:     str
    model_metadata:      ModelMetadata


class RiskEntry(BaseModel):
    date:       date
    risk_score: float


class PatientSummary(BaseModel):
    patient_id:  str
    sex:         str
    age:         float
    latest_risk: Optional[float]
    latest_date: Optional[date]


class PatientListResponse(BaseModel):
    total:    int
    limit:    int
    offset:   int
    patients: list[PatientSummary]


class PatientDetail(BaseModel):
    patient_id:   str
    sex:          str
    age:          float
    risk_history: list[RiskEntry]
    exam_count:   int
    export_path:  Optional[str]


class FLStatus(BaseModel):
    model_ready:      bool
    checkpoint_path:  Optional[str]
    rounds_completed: int
    last_updated:     Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_record(e: ExamInput) -> ExamRecord:
    try:
        phase = ClinicalPhase.from_str(e.phase)
    except ValueError:
        phase = ClinicalPhase.HOSPITALIZED
    return ExamRecord(
        exam_name=e.exam_name,
        date=e.date,
        value=e.value,
        phase=phase,
        ref_low=e.ref_low,
        ref_high=e.ref_high,
    )


async def _run_ingest(request: IngestRequest, token_fp: str) -> IngestResponse:
    engine = _get_engine()
    pid    = _pid_to_internal(request.patient_id)
    if request.output_dir:
        out = Path(request.output_dir)
        if ".." in out.parts:
            raise HTTPException(status_code=422, detail="output_dir não pode conter '..'")
        out = out.resolve()
        if _ENV == "production" and not str(out).startswith(str(_OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=422, detail="output_dir fora do diretório permitido em produção")
    else:
        out = _OUTPUT_DIR

    async with _patient_lock(pid):
        # Prepara linhas de exames (resolução canônica fora da transação — sem IO de banco)
        exam_rows = [
            {
                "analyte":            canonical,
                "date":               str(e.date),
                "value":              e.value,
                "phase":              e.phase,
                "ref_low":            e.ref_low,
                "ref_high":           e.ref_high,
                "origin":             e.origin,
                "exam_group":         e.exam_group,
                "value_text":         e.value_text,
                "unit":               e.unit,
                "attendance_id":      e.attendance_id,
                "canonical_ref_low":  canon_ref_low,
                "canonical_ref_high": canon_ref_high,
                "classification":     classification,
            }
            for e in request.exams
            for canonical, classification, canon_ref_low, canon_ref_high
            in (engine.resolve_for_ingest(e.exam_name, e.value),)
        ]

        # Transação única: persiste paciente + exames + risco + path de exportação.
        # Se qualquer passo falhar, o banco reverte completamente (sem estado parcial).
        with _db.begin() as conn:
            _db.upsert_patient_tx(conn, pid, request.sex, request.age)
            _db.add_exams_tx(conn, pid, exam_rows)

            all_exam_rows = _db.get_exams_tx(conn, pid)
            history_records = [
                ExamRecord(
                    exam_name=r["analyte"],
                    date=r["date"],
                    value=r["value"],
                    phase=ClinicalPhase.from_str(r["phase"]),
                    ref_low=r["ref_low"],
                    ref_high=r["ref_high"],
                )
                for r in all_exam_rows
            ]

            proba      = engine.predict_proba(history_records)
            risk_score = proba["risk_score"]
            today      = date.today()
            _db.add_risk_tx(conn, pid, today, risk_score)

            all_risk_rows = _db.get_risk_history_tx(conn, pid)
            patient_row   = _db.get_patient_tx(conn, pid)
            if patient_row is None:
                raise HTTPException(status_code=500, detail="Erro interno: paciente não encontrado após upsert")

            patient_export = PatientExport(
                patient_id=pid,
                sex=patient_row["sex"],
                age=patient_row["age"],
                exam_records=history_records,
                risk_predictions=[
                    RiskPrediction(date=h["date"], risk_score=h["risk_score"])
                    for h in all_risk_rows
                ],
            )
            # Exportação de arquivo: idempotente (sobrescreve). Se falhar, a transação reverte.
            export_path = _exporter.export(patient_export, out)
            _db.set_export_path_tx(conn, pid, str(export_path))

    logger.info(
        "exam_ingested",
        extra={
            "patient_id_hash": audit.pseudonymize(request.patient_id),
            "exam_count":      len(request.exams),
            "history_size":    len(history_records),
            "risk_score":      round(risk_score, 4),
            "export_path":     str(export_path),
        },
    )
    audit.log_access(
        "ingest",
        token_fp=token_fp,
        patient_id=request.patient_id,
        exam_count=len(request.exams),
        risk_score=round(risk_score, 4),
    )
    return IngestResponse(
        patient_id=pid,
        risk_score=round(risk_score, 4),
        export_path=str(export_path),
        class_probabilities={k: ClassProbability(**v) for k, v in proba["probabilities"].items()},
        predicted_class=proba["predicted_class"],
        predicted_label=proba["predicted_label"],
        model_metadata=ModelMetadata(
            trained=proba["trained"],
            calibrated=proba.get("calibrated", False),
            mc_samples=proba["mc_samples"],
            checkpoint_round=proba.get("checkpoint_round"),
            checkpoint_at=proba.get("checkpoint_at"),
            model_version=proba.get("model_version"),
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def dashboard():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "MOSAIC-FL API — painel web não encontrado em static/index.html"}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    body:    PredictRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Score de risco pontual — não persiste estado, não exporta arquivos."""
    await _rate_check(request, _api_limiter)
    records = [_to_record(e) for e in body.exams]
    if not records:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    proba = _get_engine().predict_proba(records)
    audit.log_access("predict", token_fp=fingerprint, patient_id=body.patient_id)
    return PredictResponse(
        patient_id=_pid_to_internal(body.patient_id),
        risk_score=proba["risk_score"],
        risk_date=date.today(),
        class_probabilities={k: ClassProbability(**v) for k, v in proba["probabilities"].items()},
        predicted_class=proba["predicted_class"],
        predicted_label=proba["predicted_label"],
        model_metadata=ModelMetadata(
            trained=proba["trained"],
            calibrated=proba.get("calibrated", False),
            mc_samples=proba["mc_samples"],
            checkpoint_round=proba.get("checkpoint_round"),
            checkpoint_at=proba.get("checkpoint_at"),
            model_version=proba.get("model_version"),
        ),
    )


@app.post("/api/exams/ingest", response_model=IngestResponse)
async def ingest_exams(
    request: Request,
    body:    IngestRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Ingere exames, gera score sobre histórico completo e exporta arquivos ClinicalPath."""
    await _rate_check(request, _ingest_limiter)
    if not body.exams:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    return await _run_ingest(body, token_fp=fingerprint)


@app.get("/api/patients", response_model=PatientListResponse)
async def list_patients(
    request: Request,
    fingerprint: str = Depends(_get_token_fingerprint),
    limit:  int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0,   ge=0),
):
    await _rate_check(request, _api_limiter)
    audit.log_access("patient_list", token_fp=fingerprint)
    total = _db.count_patients()
    rows  = _db.list_patients(limit=limit, offset=offset)
    return PatientListResponse(
        total=total,
        limit=limit,
        offset=offset,
        patients=[
            PatientSummary(
                patient_id=r["patient_id"],
                sex=r["sex"],
                age=r["age"],
                latest_risk=round(r["latest_risk"], 4) if r["latest_risk"] is not None else None,
                latest_date=(
                    r["latest_date"] if isinstance(r["latest_date"], date)
                    else date.fromisoformat(r["latest_date"])
                ) if r["latest_date"] else None,
            )
            for r in rows
        ],
    )


@app.get("/api/patients/{patient_id}", response_model=PatientDetail)
async def get_patient(
    request: Request,
    patient_id: str,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    await _rate_check(request, _api_limiter)
    pid = _pid_to_internal(patient_id)
    audit.log_access("patient_read", token_fp=fingerprint, patient_id=patient_id)

    p = _db.get_patient(pid)
    if p is None:
        raise HTTPException(status_code=404, detail=f"paciente não encontrado")

    risk_history = _db.get_risk_history(pid)
    return PatientDetail(
        patient_id=pid,
        sex=p["sex"],
        age=p["age"],
        risk_history=[
            RiskEntry(date=h["date"], risk_score=h["risk_score"])
            for h in risk_history
        ],
        exam_count=_db.exam_count(pid),
        export_path=_db.get_export_path(pid),
    )


@app.get("/api/fl/status", response_model=FLStatus)
async def fl_status():
    ckpt = _latest_checkpoint()
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


@app.post("/api/fl/reload")
async def reload_model(fingerprint: str = Depends(_get_token_fingerprint)):
    """Força recarga do checkpoint mais recente com verificação de integridade."""
    global _engine
    ckpt = _latest_checkpoint()
    if not ckpt:
        raise HTTPException(status_code=404, detail="nenhum checkpoint disponível")

    if not _verify_checkpoint_integrity(ckpt):
        logger.error("checkpoint_integrity_failed path=%s", ckpt)
        raise HTTPException(
            status_code=500,
            detail=f"Checkpoint corrompido ou adulterado: {ckpt.name}. Retrainamento necessário.",
        )

    _engine = InferenceEngine(
        checkpoint_path=ckpt,
        db_url=os.getenv("FL_DB_URL"),
    )
    logger.info("model_reloaded", extra={"checkpoint": str(ckpt)})
    audit.log_access("model_reload", token_fp=fingerprint, checkpoint=str(ckpt))
    return {"reloaded": True, "checkpoint": str(ckpt)}
