"""
service.py
FastAPI — endpoint de inferência, ingestão de exames e painel web.

Autenticação: aceita qualquer token presente no header X-API-Key ou
              Authorization: Bearer <token>. A validação de identidade
              é responsabilidade do IAM hospitalar upstream; este módulo
              apenas exige presença de token (FL_AUTH_REQUIRED=true, padrão)
              e registra um fingerprint SHA-256 truncado para rastreabilidade
              LGPD Art. 37.

              FL_AUTH_REQUIRED=false desativa a exigência (dev/testes).

CORS: configurado via FL_CORS_ORIGINS (padrão: * para desenvolvimento).

Persistência: SQLite via db.PatientDB — histórico de pacientes sobrevive a reinicios.

Auditoria: todo acesso a dado de paciente é registrado em logs/audit.log
           com patient_id pseudonimizado e fingerprint do token.

Endpoints:
  POST /api/predict             — score de risco (sem persistir estado)
  POST /api/exams/ingest        — ingere + prediz + exporta ClinicalPath
  GET  /api/patients            — lista de pacientes com último score
  GET  /api/patients/{id}       — histórico completo + contagem de exames
  GET  /api/fl/status           — status do modelo (checkpoint, rounds)
  POST /api/fl/reload           — recarrega checkpoint mais recente
  GET  /                        — painel web
"""
import asyncio
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
_OUTPUT_DIR = Path(os.getenv("FL_CLINICALPATH_OUTPUT", "data/clinicalpath_output"))
_CORS_ORIGINS = os.getenv("FL_CORS_ORIGINS", "*").split(",")
_AUTH_REQUIRED = os.getenv("FL_AUTH_REQUIRED", "true").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_db = PatientDB()
_exporter = ClinicalPathExporter()
_engine: Optional[InferenceEngine] = None
_ingest_lock = asyncio.Lock()


def _latest_checkpoint() -> Optional[Path]:
    if not _CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(_CHECKPOINT_DIR.glob("round_*.pt"))
    return ckpts[-1] if ckpts else None


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(checkpoint_path=_latest_checkpoint())
    return _engine


# ---------------------------------------------------------------------------
# Autenticação + fingerprinting (LGPD Art. 37)
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_header = HTTPBearer(auto_error=False)


async def _get_token_fingerprint(
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_header),
) -> str:
    """
    Extrai e valida presença de token; retorna fingerprint SHA-256 (12 hex chars).

    Aceita X-API-Key ou Authorization: Bearer <token>.
    Não valida o conteúdo do token — isso é responsabilidade do IAM upstream.
    FL_AUTH_REQUIRED=false desativa a exigência (modo dev/testes).
    """
    token = api_key or (bearer.credentials if bearer else None)
    if _AUTH_REQUIRED and not token:
        raise HTTPException(status_code=403, detail="Token de autorização ausente")
    if not token:
        return "dev-mode"
    return audit.token_fingerprint(token)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MOSAIC-FL API",
    description="Inferência federada e painel de monitoramento clínico",
    version="0.2.0",
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
    exam_name: str
    date: date
    value: float
    phase: str = Field("IN", description="AB | EX | IN | OBITO | P_ALTA")
    ref_low: float = 0.0
    ref_high: float = 0.0
    sex_ref_low: float = 0.0
    sex_ref_high: float = 0.0


class PredictRequest(BaseModel):
    patient_id: str
    exams: list[ExamInput]


class PredictResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_date: date


class IngestRequest(BaseModel):
    patient_id: str
    sex: str = "M"
    age: float = 0.0
    exams: list[ExamInput]
    output_dir: Optional[str] = None


class IngestResponse(BaseModel):
    patient_id: str
    risk_score: float
    export_path: str


class RiskEntry(BaseModel):
    date: date
    risk_score: float


class PatientSummary(BaseModel):
    patient_id: str
    sex: str
    age: float
    latest_risk: Optional[float]
    latest_date: Optional[date]


class PatientDetail(BaseModel):
    patient_id: str
    sex: str
    age: float
    risk_history: list[RiskEntry]
    exam_count: int
    export_path: Optional[str]


class FLStatus(BaseModel):
    model_ready: bool
    checkpoint_path: Optional[str]
    rounds_completed: int
    last_updated: Optional[str]


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
        sex_ref_low=e.sex_ref_low,
        sex_ref_high=e.sex_ref_high,
    )


async def _run_ingest(request: IngestRequest, token_fp: str) -> IngestResponse:
    records = [_to_record(e) for e in request.exams]
    engine = _get_engine()
    risk_score = engine.predict(records)
    today = date.today()
    out = Path(request.output_dir) if request.output_dir else _OUTPUT_DIR

    async with _ingest_lock:
        _db.upsert_patient(request.patient_id, request.sex, request.age)
        _db.add_exams(request.patient_id, [e.model_dump() | {"date": str(e.date)} for e in request.exams])
        _db.add_risk(request.patient_id, today.isoformat(), risk_score)

        all_exam_rows = _db.get_exams(request.patient_id)
        all_risk_rows = _db.get_risk_history(request.patient_id)

        all_records = [
            ExamRecord(
                exam_name=r["exam_name"],
                date=date.fromisoformat(r["date"]),
                value=r["value"],
                phase=ClinicalPhase.from_str(r["phase"]),
                ref_low=r["ref_low"],
                ref_high=r["ref_high"],
                sex_ref_low=r["sex_ref_low"],
                sex_ref_high=r["sex_ref_high"],
            )
            for r in all_exam_rows
        ]
        patient_row = next(
            p for p in _db.list_patients() if p["patient_id"] == request.patient_id
        )
        patient_export = PatientExport(
            patient_id=request.patient_id,
            sex=patient_row["sex"],
            age=patient_row["age"],
            exam_records=all_records,
            risk_predictions=[
                RiskPrediction(date=date.fromisoformat(h["date"]), risk_score=h["risk_score"])
                for h in all_risk_rows
            ],
        )
        export_path = _exporter.export(patient_export, out)
        _db.set_export_path(request.patient_id, str(export_path))

    logger.info(
        "exam_ingested",
        extra={
            "patient_id_hash": audit.pseudonymize(request.patient_id),
            "exam_count": len(records),
            "risk_score": round(risk_score, 4),
            "export_path": str(export_path),
        },
    )
    audit.log_access(
        "ingest",
        token_fp=token_fp,
        patient_id=request.patient_id,
        exam_count=len(records),
        risk_score=round(risk_score, 4),
    )
    return IngestResponse(
        patient_id=request.patient_id,
        risk_score=round(risk_score, 4),
        export_path=str(export_path),
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
    body: PredictRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Score de risco pontual — não persiste estado, não exporta arquivos."""
    records = [_to_record(e) for e in body.exams]
    if not records:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    risk_score = _get_engine().predict(records)
    audit.log_access("predict", token_fp=fingerprint, patient_id=body.patient_id)
    return PredictResponse(
        patient_id=body.patient_id,
        risk_score=round(risk_score, 4),
        risk_date=date.today(),
    )


@app.post("/api/exams/ingest", response_model=IngestResponse)
async def ingest_exams(
    body: IngestRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Ingere exames, gera score e exporta arquivos ClinicalPath."""
    if not body.exams:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    return await _run_ingest(body, token_fp=fingerprint)


@app.get("/api/patients", response_model=list[PatientSummary])
async def list_patients(fingerprint: str = Depends(_get_token_fingerprint)):
    audit.log_access("patient_list", token_fp=fingerprint)
    rows = _db.list_patients()
    return [
        PatientSummary(
            patient_id=r["patient_id"],
            sex=r["sex"],
            age=r["age"],
            latest_risk=round(r["latest_risk"], 4) if r["latest_risk"] is not None else None,
            latest_date=date.fromisoformat(r["latest_date"]) if r["latest_date"] else None,
        )
        for r in rows
    ]


@app.get("/api/patients/{patient_id}", response_model=PatientDetail)
async def get_patient(
    patient_id: str,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    audit.log_access("patient_read", token_fp=fingerprint, patient_id=patient_id)
    if not _db.patient_exists(patient_id):
        raise HTTPException(status_code=404, detail=f"paciente {patient_id!r} não encontrado")

    patients = {p["patient_id"]: p for p in _db.list_patients()}
    p = patients[patient_id]
    risk_history = _db.get_risk_history(patient_id)
    return PatientDetail(
        patient_id=patient_id,
        sex=p["sex"],
        age=p["age"],
        risk_history=[
            RiskEntry(date=date.fromisoformat(h["date"]), risk_score=h["risk_score"])
            for h in risk_history
        ],
        exam_count=_db.exam_count(patient_id),
        export_path=_db.get_export_path(patient_id),
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
    """Força recarga do checkpoint mais recente (chamado automaticamente após round FL)."""
    global _engine
    ckpt = _latest_checkpoint()
    if not ckpt:
        raise HTTPException(status_code=404, detail="nenhum checkpoint disponível")
    _engine = InferenceEngine(checkpoint_path=ckpt)
    logger.info("model_reloaded", extra={"checkpoint": str(ckpt)})
    audit.log_access("model_reload", token_fp=fingerprint, checkpoint=str(ckpt))
    return {"reloaded": True, "checkpoint": str(ckpt)}
