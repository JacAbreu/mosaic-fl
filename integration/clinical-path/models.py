from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class ClinicalPhase(Enum):
    """Phases of patient care, each mapped to the ClinicalPath status string and integer code."""

    OUTPATIENT = ("AB", -2)
    PRE_HOSPITAL = ("EX", -1)
    HOSPITALIZED = ("IN", 0)
    DEATH = ("OBITO", 0)
    POST_DISCHARGE = ("P_ALTA", 1)

    @property
    def status_str(self) -> str:
        return self.value[0]

    @property
    def status_code(self) -> int:
        return self.value[1]

    @classmethod
    def from_str(cls, s: str) -> "ClinicalPhase":
        for member in cls:
            if member.status_str == s:
                return member
        raise ValueError(f"fase clínica desconhecida: {s!r}")


@dataclass
class ExamRecord:
    """A single lab or clinical measurement for a patient on a given day."""

    exam_name: str
    date: date
    value: float
    phase: ClinicalPhase
    ref_low: float = 0.0
    ref_high: float = 0.0
    sex_ref_low: float = 0.0
    sex_ref_high: float = 0.0


@dataclass
class ProbabilityEstimate:
    """Probability and MC-Dropout uncertainty for one clinical evolution class."""

    value: float        # mean probability across MC samples ∈ [0, 1]
    uncertainty: float  # std across MC samples ∈ [0, 1]


@dataclass
class RiskPrediction:
    """MOSAIC-FL prediction for a patient on a given day.

    risk_score is a scalar summary derived from class_probabilities:
        risk_score = Σ(p_i × w_i),  w_i = linspace(0, 1, n_classes)
    It is never computed independently — always a function of the distribution.

    class_probabilities is empty for historical entries loaded from the database
    (only risk_score is persisted). The current prediction always carries the
    full distribution.
    """

    date: date
    risk_score: float
    phase: ClinicalPhase = ClinicalPhase.HOSPITALIZED
    class_probabilities: dict[str, ProbabilityEstimate] = field(default_factory=dict)


@dataclass
class PatientExport:
    """All data needed to write ClinicalPath files for one patient."""

    patient_id: str
    sex: str  # "M" or "F"
    age: float
    exam_records: list[ExamRecord] = field(default_factory=list)
    risk_predictions: list[RiskPrediction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic exam constants — FL_RISK_SCORE
# ---------------------------------------------------------------------------

# Scalar risk summary ∈ [0,1]. ref_high=0.3: values above 0.3 render as
# abnormal in ClinicalPath's colour scale.
FL_RISK_EXAM_NAME = "FL_RISK_SCORE"
FL_RISK_REF_LOW: float = 0.0
FL_RISK_REF_HIGH: float = 0.3

# ---------------------------------------------------------------------------
# Synthetic exam constants — class probability distribution
# ---------------------------------------------------------------------------

# Exam name pattern: FL_PROB_{LABEL} and FL_PROB_{LABEL}_INCERTEZA
# Both use ref_low=0.0, ref_high=1.0 (valid probability range).
# Uncertainty ref_high=0.15: MC-Dropout std above 15% signals low model
# confidence and should be visible in ClinicalPath's colour scale.
FL_PROB_EXAM_PREFIX         = "FL_PROB_"
FL_PROB_UNCERTAINTY_SUFFIX  = "_INCERTEZA"
FL_PROB_REF_LOW: float      = 0.0
FL_PROB_REF_HIGH: float     = 1.0
FL_PROB_UNCERTAINTY_REF_HIGH: float = 0.15


def prob_exam_name(label: str) -> str:
    """Returns the synthetic exam name for a class probability value."""
    return f"{FL_PROB_EXAM_PREFIX}{label.upper()}"


def uncertainty_exam_name(label: str) -> str:
    """Returns the synthetic exam name for a class probability uncertainty."""
    return f"{FL_PROB_EXAM_PREFIX}{label.upper()}{FL_PROB_UNCERTAINTY_SUFFIX}"
