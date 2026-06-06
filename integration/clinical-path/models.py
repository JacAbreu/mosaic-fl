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
class RiskPrediction:
    """MOSAIC-FL risk score for a patient on a given day (0.0–1.0)."""

    date: date
    risk_score: float
    phase: ClinicalPhase = ClinicalPhase.HOSPITALIZED


@dataclass
class PatientExport:
    """All data needed to write ClinicalPath files for one patient."""

    patient_id: str
    sex: str  # "M" or "F"
    age: float
    exam_records: list[ExamRecord] = field(default_factory=list)
    risk_predictions: list[RiskPrediction] = field(default_factory=list)


# Synthetic exam injected by MOSAIC-FL.
# ref_high=0.3: scores above 0.3 are rendered as abnormal by ClinicalPath's color scale.
FL_RISK_EXAM_NAME = "FL_RISK_SCORE"
FL_RISK_REF_LOW: float = 0.0
FL_RISK_REF_HIGH: float = 0.3
