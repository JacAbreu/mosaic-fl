"""Pydantic request/response schemas — MOSAIC-FL API."""
import math
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExamInput(BaseModel):
    exam_name:     str
    date:          date
    value:         float
    phase:         str   = Field("IN", description="AB | EX | IN | OBITO | P_ALTA")
    ref_low:       float = 0.0
    ref_high:      float = 0.0
    origin:        Optional[str] = None
    exam_group:    Optional[str] = None
    value_text:    Optional[str] = None
    unit:          Optional[str] = None
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
    patient_id:        str
    sex:               str   = "M"
    age:               float = 0.0
    exams:             list[ExamInput]
    output_dir:        Optional[str] = None
    correlation_token: Optional[str] = None


class IngestResponse(BaseModel):
    patient_id:           str
    risk_score:           float
    export_path:          str
    class_probabilities:  dict[str, ClassProbability]
    predicted_class:      int
    predicted_label:      str
    model_metadata:       ModelMetadata
    fhir_risk_assessment: Optional[dict] = None


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


class OutcomeFeedbackRequest(BaseModel):
    correlation_token: str
    actual_outcome:    str = Field(
        ...,
        description="Desfecho real na alta (ex: 'alta', 'obito', 'internacao_prolongada')",
    )
    source: str = Field(
        default="manual",
        description="Origem do registro: 'manual', 'epr', 'fhir'",
    )


class OutcomeFeedbackResponse(BaseModel):
    recorded:          bool
    correlation_token: str
    predicted_label:   Optional[str] = None
