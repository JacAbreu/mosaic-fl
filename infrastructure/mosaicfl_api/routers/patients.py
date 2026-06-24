"""Endpoints de pacientes: /api/patients, /api/patients/{id}, /api/patients/{id}/outcome."""
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import audit
from .. import state
from ..schemas import (
    OutcomeFeedbackRequest, OutcomeFeedbackResponse,
    PatientDetail, PatientListResponse, PatientSummary, RiskEntry,
)
from ..security import _api_limiter, _get_token_fingerprint, _pid_to_internal, _rate_check

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/patients", response_model=PatientListResponse)
async def list_patients(
    request:     Request,
    fingerprint: str = Depends(_get_token_fingerprint),
    limit:       int = Query(default=100, ge=1, le=500),
    offset:      int = Query(default=0,   ge=0),
):
    await _rate_check(request, _api_limiter)
    audit.log_access("patient_list", token_fp=fingerprint)
    total = state._db.count_patients()
    rows  = state._db.list_patients(limit=limit, offset=offset)
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


@router.get("/api/patients/{patient_id}", response_model=PatientDetail)
async def get_patient(
    request:    Request,
    patient_id: str,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    await _rate_check(request, _api_limiter)
    pid = _pid_to_internal(patient_id)
    audit.log_access("patient_read", token_fp=fingerprint, patient_id=patient_id)

    p = state._db.get_patient(pid)
    if p is None:
        raise HTTPException(status_code=404, detail="paciente não encontrado")

    risk_history = state._db.get_risk_history(pid)
    return PatientDetail(
        patient_id=pid,
        sex=p["sex"],
        age=p["age"],
        risk_history=[RiskEntry(date=h["date"], risk_score=h["risk_score"]) for h in risk_history],
        exam_count=state._db.exam_count(pid),
        export_path=state._db.get_export_path(pid),
    )


@router.post("/api/patients/{patient_id}/outcome", response_model=OutcomeFeedbackResponse)
async def record_outcome(
    request:     Request,
    patient_id:  str,
    body:        OutcomeFeedbackRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """
    Registra desfecho real observado na alta.

    Chamado pelo hospital (manualmente ou via integração EPR/FHIR) após a alta do paciente.
    O correlation_token vincula o desfecho à predição feita durante a internação sem expor
    identidade do paciente ao servidor FL.
    """
    await _rate_check(request, _api_limiter)

    if not state._db.prediction_exists(body.correlation_token):
        raise HTTPException(
            status_code=404,
            detail="correlation_token não encontrado — verifique se a predição foi registrada via /api/exams/ingest",
        )

    recorded = state._db.record_outcome(
        correlation_token=body.correlation_token,
        actual_label=body.actual_outcome,
        source=body.source,
    )

    audit.log_access(
        "outcome_recorded",
        token_fp=fingerprint,
        patient_id=patient_id,
        correlation_token=body.correlation_token,
        actual_outcome=body.actual_outcome,
        source=body.source,
        recorded=recorded,
    )
    logger.info(
        "outcome_recorded token=%s outcome=%s source=%s recorded=%s",
        body.correlation_token, body.actual_outcome, body.source, recorded,
    )
    return OutcomeFeedbackResponse(
        recorded=recorded,
        correlation_token=body.correlation_token,
    )
