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
    calibration_method: str           = "temperature"  # "temperature" | "isotonic" — ver FL_CALIBRATION_METHOD
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


class RagExplanation(BaseModel):
    justificativa:        Optional[str]  = None
    fontes:                list[dict]    = []
    alucinacao_detectada: bool           = False
    confiavel:            bool           = False
    llm_backend:          Optional[str]  = None
    llm_model_used:       Optional[str]  = None
    llm_was_fallback:     bool           = False
    erro:                 Optional[str]  = None  # preenchido quando a explicação não pôde ser gerada


class PredictResponse(BaseModel):
    patient_id:          str
    risk_score:          float
    risk_date:           date
    class_probabilities: dict[str, ClassProbability]
    predicted_class:     int
    predicted_label:     str
    model_metadata:      ModelMetadata
    rag_explanation:     Optional[RagExplanation] = None


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


class VocabAnomalyEntry(BaseModel):
    canonical:    str
    source:       str
    active:       bool
    created_at:   Optional[str] = None
    reason:       str = Field(description="Por que foi sinalizado — heurística best-effort, não prova de erro")
    ref_low:      Optional[float] = None
    ref_high:     Optional[float] = None
    n_hospitals:  Optional[int] = None
    local_record_count: int = Field(
        description="Registros com este analito no banco que esta instância da API enxerga — "
                     "NÃO é o total federado, só reflete o hospital local"
    )


class VocabAnomalyListResponse(BaseModel):
    anomalies: list[VocabAnomalyEntry]


class VocabAnomalyActionRequest(BaseModel):
    active: bool


class VocabAnomalyActionResponse(BaseModel):
    canonical: str
    active:    bool


class VocabAnomalyCorrectionRequest(BaseModel):
    correct_canonical: str = Field(description="Nome canônico correto (ex: '183' -> 'AMILASE')")


class VocabAnomalyCorrectionResponse(BaseModel):
    old_canonical:     str
    new_canonical:     str
    merged:            bool = Field(description="True se new_canonical já existia (juntou com ele); "
                                                  "False se foi um rename simples")
    exam_records_updated: int


class OrchestrationConfigResponse(BaseModel):
    """Peso de classe explícito (cost-sensitive learning) — ver mosaicfl.core.class_weighting
    e docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14. Lido de
    clinical.fl_orchestration_config (migration 028), o mesmo canal que já entrega
    proximal_mu/stop pro cliente a cada rodada — idêntico nos dois hospitais."""
    class_weight_overrides: dict[str, float] = Field(
        description="Classe presente aqui usa peso explícito; ausente cai em class_balanced "
                     "(frequência local, comportamento padrão do projeto)"
    )
    class_labels: list[str] = Field(description="Todas as classes de prognóstico configuradas (FL_CLASS_LABELS)")
    class_weight_clamp: float = Field(description="Teto de estabilidade aplicado a qualquer peso, mesmo explícito")


class ClassWeightOverridesUpdate(BaseModel):
    overrides: dict[str, float] = Field(
        description="Substitui o conjunto inteiro de overrides (não é merge parcial). "
                     "Objeto vazio remove todos os overrides — todas as classes voltam pra class_balanced."
    )


class TrainingResultSummary(BaseModel):
    """1 linha de metrics.fl_trainings — visão resumida pra tela /fl-training-results."""
    id: int
    algorithm: Optional[str] = None
    run_classification: Optional[str] = None
    partition_mode: Optional[str] = None
    status: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    n_rounds_done: Optional[int] = None
    best_round: Optional[int] = None
    best_accuracy: Optional[float] = None
    converged: Optional[bool] = None
    macro_f1: Optional[float] = None
    macro_auc: Optional[float] = None
    ece: Optional[float] = None
    ece_pre: Optional[float] = None
    total_duration_s: Optional[float] = None
    dp_noise_multiplier: Optional[float] = None
    dp_noise_strategy: Optional[str] = None
    dp_epsilon_simple: Optional[float] = None
    dp_epsilon_rdp: Optional[float] = None
    is_active_model: Optional[bool] = None
    checkpoint_round: Optional[int] = None
    checkpoint_mismatch: Optional[bool] = None


class TrainingResultsListResponse(BaseModel):
    trainings: list[TrainingResultSummary]


class TrainingRoundDetail(BaseModel):
    """1 linha de metrics.fl_round_history — pra expansão por treino."""
    round: int
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    per_class_f1: Optional[list[float]] = None
    per_client_f1: Optional[list[dict]] = None


class TrainingRoundsResponse(BaseModel):
    training_id: int
    class_labels: list[str]
    best_round: Optional[int] = None
    checkpoint_round: Optional[int] = None
    evaluation_json: Optional[dict] = None
    rounds: list[TrainingRoundDetail]


class TrainingComparisonSide(BaseModel):
    """1 lado da comparação latest vs. best — resumo + per_class_f1 no melhor round."""
    training: TrainingResultSummary
    per_class_f1: Optional[list[float]] = None


class TrainingComparisonResponse(BaseModel):
    class_labels: list[str]
    latest: Optional[TrainingComparisonSide] = None
    best: Optional[TrainingComparisonSide] = None


class RagEvaluationEntry(BaseModel):
    """1 linha de metrics.rag_likert_evaluations. likert_score é a nota HUMANA
    (None = ainda pendente); llm_judge_score é automática, complementar, nunca
    substitui a humana (achado 2026-07-29)."""
    id: int
    patient_id_hash: str
    predicted_label: str
    risk_score: float
    justificativa: str
    fontes_json: Optional[list] = None
    llm_backend: Optional[str] = None
    llm_model_used: Optional[str] = None
    llm_was_fallback: bool = False
    alucinacao_detectada: bool = False
    confiavel: bool = False
    likert_score: Optional[int] = None
    llm_judge_score: Optional[int] = None
    llm_judge_rationale: Optional[str] = None
    evaluator: Optional[str] = None
    checkpoint_round: Optional[int] = None
    created_at: Optional[str] = None


class RagEvaluationListResponse(BaseModel):
    evaluations: list[RagEvaluationEntry]


class RagEvaluationScoreRequest(BaseModel):
    likert_score: int = Field(ge=1, le=5)
    evaluator: str


class RagEvaluationReportResponse(BaseModel):
    n_total: int
    n_hallucination: int
    n_confiavel: int
    n_human_scored: int
    human_pct_ge4: Optional[float] = None
    human_distribution: dict[int, int] = {}
    n_judge_scored: int
    judge_pct_ge4: Optional[float] = None
    judge_distribution: dict[int, int] = {}
    n_both_scored: int
    agreement_exact_pct: Optional[float] = None
    agreement_ge4_pct: Optional[float] = None
