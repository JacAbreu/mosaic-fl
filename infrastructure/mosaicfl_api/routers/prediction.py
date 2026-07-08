"""Endpoints de predição: /api/predict e /api/exams/ingest."""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import audit
from .. import state
from ..schemas import (
    ClassProbability, ExamInput, IngestRequest, IngestResponse,
    ModelMetadata, PredictRequest, PredictResponse, RagExplanation,
)
from ..security import (
    _api_limiter, _get_token_fingerprint, _ingest_limiter,
    _pid_to_internal, _rate_check,
)
from ..state import ClinicalPhase, ExamRecord, InferenceOutput, PatientExport, ProbabilityEstimate, RiskPrediction

logger = logging.getLogger(__name__)

router = APIRouter()


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
    engine = state._get_engine()
    pid    = _pid_to_internal(request.patient_id)
    if request.output_dir:
        out = Path(request.output_dir)
        if ".." in out.parts:
            raise HTTPException(status_code=422, detail="output_dir não pode conter '..'")
        out = out.resolve()
        _env = __import__("os").getenv("FL_ENV", "development").lower()
        if _env == "production" and not str(out).startswith(str(state._OUTPUT_DIR.resolve())):
            raise HTTPException(
                status_code=422,
                detail="output_dir fora do diretório permitido em produção",
            )
    else:
        out = state._OUTPUT_DIR

    async with state._patient_lock(pid):
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

        with state._db.begin() as conn:
            state._db.upsert_patient_tx(conn, pid, request.sex, request.age)
            state._db.add_exams_tx(conn, pid, exam_rows)

            all_exam_rows = state._db.get_exams_tx(conn, pid)
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
            state._db.add_risk_tx(conn, pid, today, risk_score)

            if request.correlation_token:
                state._db.store_prediction_tx(
                    conn,
                    correlation_token=request.correlation_token,
                    patient_id_hash=audit.pseudonymize(request.patient_id),
                    predicted_class=proba["predicted_class"],
                    predicted_label=proba["predicted_label"],
                    class_probabilities=proba["probabilities"],
                    risk_score=risk_score,
                    model_round=proba.get("checkpoint_round"),
                    model_version=proba.get("model_version"),
                )

            all_risk_rows = state._db.get_risk_history_tx(conn, pid)
            patient_row   = state._db.get_patient_tx(conn, pid)
            if patient_row is None:
                raise HTTPException(
                    status_code=500,
                    detail="Erro interno: paciente não encontrado após upsert",
                )

            today_proba = {
                k: ProbabilityEstimate(value=v["value"], uncertainty=v["uncertainty"])
                for k, v in proba["probabilities"].items()
            }
            patient_export = PatientExport(
                patient_id=pid,
                sex=patient_row["sex"],
                age=patient_row["age"],
                exam_records=history_records,
                risk_predictions=[
                    RiskPrediction(
                        date=h["date"],
                        risk_score=h["risk_score"],
                        class_probabilities=today_proba if h["date"] == today else {},
                    )
                    for h in all_risk_rows
                ],
            )
            export_path = state._exporter.export(patient_export, out)
            state._db.set_export_path_tx(conn, pid, str(export_path))

            fhir_output = InferenceOutput(
                predictions=[(k, v["value"]) for k, v in proba["probabilities"].items()],
                model_round=proba.get("checkpoint_round") or 0,
                temperature=proba.get("temperature", 1.0),
                ece=proba.get("ece", 0.0),
                correlation_token=request.correlation_token or "",
            )
            fhir_ra = state._fhir_exporter.to_risk_assessment(fhir_output)

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
        fhir_risk_assessment=fhir_ra,
        model_metadata=ModelMetadata(
            trained=proba["trained"],
            calibrated=proba.get("calibrated", False),
            mc_samples=proba["mc_samples"],
            checkpoint_round=proba.get("checkpoint_round"),
            checkpoint_at=proba.get("checkpoint_at"),
            model_version=proba.get("model_version"),
        ),
    )


def _build_rag_explanation(records: list[ExamRecord], proba: dict) -> RagExplanation:
    """Gera a justificativa via RAG. Nunca propaga exceção — RAG é enriquecimento
    opcional da predição (Ollama fora do ar, timeout etc. não devem derrubar
    /api/predict); falhas viram RagExplanation com `erro` preenchido."""
    try:
        tokens = ", ".join(f"{r.exam_name}={r.value}" for r in records)
        predicted_label = proba["predicted_label"]
        probability = proba["probabilities"][predicted_label]["value"]
        rag = state._get_rag()
        result = rag.explain(
            patient_data={"tokens": tokens},
            model_prediction={"diagnostico": predicted_label, "probabilidade": probability},
        )
        return RagExplanation(
            justificativa=result["justificativa"],
            fontes=result["fontes"],
            alucinacao_detectada=result["alucinacao_detectada"],
            confiavel=result["confiavel"],
            llm_backend=result["llm_backend"],
            llm_model_used=result["llm_model_used"],
            llm_was_fallback=result["llm_was_fallback"],
        )
    except Exception as exc:
        logger.warning("rag_explanation_failed: %s", exc)
        return RagExplanation(erro="Não foi possível gerar a explicação via RAG neste momento.")


@router.post("/api/predict", response_model=PredictResponse)
async def predict(
    request:     Request,
    body:        PredictRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
    explain:     bool = True,
):
    """Score de risco pontual — não persiste estado, não exporta arquivos.

    explain=True (padrão): inclui justificativa gerada via RAG na resposta —
    mais lento (chamada ao LLM). Use ?explain=false para resposta rápida, sem
    justificativa, quando a latência do RAG não for aceitável.
    """
    await _rate_check(request, _api_limiter)
    records = [_to_record(e) for e in body.exams]
    if not records:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    proba = state._get_engine().predict_proba(records)
    audit.log_access("predict", token_fp=fingerprint, patient_id=body.patient_id)
    rag_explanation = _build_rag_explanation(records, proba) if explain else None
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
        rag_explanation=rag_explanation,
    )


@router.post("/api/exams/ingest", response_model=IngestResponse)
async def ingest_exams(
    request:     Request,
    body:        IngestRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Ingere exames, gera score sobre histórico completo e exporta arquivos ClinicalPath."""
    await _rate_check(request, _ingest_limiter)
    if not body.exams:
        raise HTTPException(status_code=422, detail="exams não pode ser vazio")
    return await _run_ingest(body, token_fp=fingerprint)
