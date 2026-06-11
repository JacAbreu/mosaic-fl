"""
outcomes_extract.py
Loads a FAPESP outcomes/desfechos CSV into:
  - clinical.attendances   (attendance record per visit)
  - metrics.clinical_outcomes (outcome per patient)

Only HSL and BPSP provide outcome data in the FAPESP dataset.

Expected source columns (resolved semantically, case-insensitive):
  patient_id, outcome_date, outcome_text
  Optional: attendance_id, attended_at, attendance_type, specialty
"""
import logging
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from integration.column_resolver import CLINICAL_SEMANTIC_MAP, ColumnResolver
from integration.fapesp.transforms import classify_outcome, normalize_optional, parse_date
from infrastructure.mosaicfl_api.db import PatientDB

logger = logging.getLogger(__name__)

_REQUIRED = {"patient_id", "outcome_date", "outcome_text"}
_RESOLVER = ColumnResolver(CLINICAL_SEMANTIC_MAP, required=_REQUIRED)


def load_outcomes(
    file: TextIOWrapper,
    db: PatientDB,
    hospital_id: str,
    chunk_size: int = 10_000,
) -> int:
    """
    Reads an outcomes CSV stream and writes to clinical.attendances and
    metrics.clinical_outcomes. Returns the number of outcome rows inserted.
    """
    total = 0
    skipped_fk = 0
    mapping: Optional[dict] = None

    for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=chunk_size,
                              encoding="utf-8", on_bad_lines="warn"):
        if mapping is None:
            mapping = _RESOLVER.resolve(list(chunk.columns))
            logger.info("outcomes_columns_resolved hospital=%s mapping=%s", hospital_id, mapping)

        for _, row in chunk.iterrows():
            patient_id = _get(row, mapping, "patient_id")
            if not patient_id:
                continue
            patient_id = patient_id.strip()

            attendance_id        = _get(row, mapping, "attendance_id")
            attended_at          = parse_date(_get(row, mapping, "attended_at"))
            attendance_type      = normalize_optional(_get(row, mapping, "attendance_type"))
            specialty            = normalize_optional(_get(row, mapping, "specialty"))
            clinic_id            = normalize_optional(_get(row, mapping, "clinic_id"))
            suspected_diagnosis  = normalize_optional(_get(row, mapping, "suspected_diagnosis"))
            confirmed_diagnosis  = normalize_optional(_get(row, mapping, "confirmed_diagnosis"))

            # Upsert attendance — skip if patient_id not in patients (FK inconsistency in
            # FAPESP dataset: some desfechos reference patients absent from the patients file)
            if attendance_id and attended_at:
                try:
                    db.upsert_attendance(
                        attendance_id       = attendance_id.strip(),
                        patient_id          = patient_id,
                        hospital_id         = hospital_id,
                        attended_at         = attended_at.isoformat(),
                        attendance_type     = attendance_type,
                        specialty           = specialty,
                        clinic_id           = clinic_id,
                        suspected_diagnosis = suspected_diagnosis,
                        confirmed_diagnosis = confirmed_diagnosis,
                    )
                except IntegrityError:
                    skipped_fk += 1
                    attendance_id = None  # don't reference the skipped attendance in outcome

            outcome_date = parse_date(_get(row, mapping, "outcome_date"))
            if not outcome_date:
                continue

            outcome_text = _get(row, mapping, "outcome_text") or ""

            db.add_clinical_outcome(
                patient_id    = patient_id,
                outcome_at    = outcome_date.isoformat(),
                outcome_text  = outcome_text.strip(),
                outcome_class = classify_outcome(outcome_text),
                attendance_id = attendance_id,
            )
            total += 1

    if skipped_fk:
        logger.warning("outcomes_skipped_attendance hospital=%s skipped=%d (patient not in patients table)", hospital_id, skipped_fk)

    logger.info("outcomes_loaded hospital=%s total=%d", hospital_id, total)
    return total


def _get(row, mapping: dict, concept: str) -> Optional[str]:
    col = mapping.get(concept)
    if col is None:
        return None
    val = row.get(col)
    if val is None or isinstance(val, float):
        return None
    return str(val).strip() or None
