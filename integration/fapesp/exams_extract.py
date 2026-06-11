"""
exams_adapter.py
Loads a FAPESP exams CSV into metrics.exam_records.

Expected source columns (resolved semantically, case-insensitive):
  patient_id, collection_date, analyte, result_text
  Optional: attendance_id, origin, exam_group, result_num, unit, reference_range
"""
import logging
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from integration.column_resolver import CLINICAL_SEMANTIC_MAP, ColumnResolver
from integration.fapesp.transforms import (
    extract_numeric,
    infer_phase,
    normalize_optional,
    parse_date,
    parse_reference_range,
)
from infrastructure.mosaicfl_api.db import PatientDB

logger = logging.getLogger(__name__)

_REQUIRED = {"patient_id", "collection_date", "analyte", "result_text"}
_RESOLVER = ColumnResolver(CLINICAL_SEMANTIC_MAP, required=_REQUIRED)

_BATCH_SIZE = 500


def load_exams(
    file: TextIOWrapper,
    db: PatientDB,
    hospital_id: str,
    chunk_size: int = 50_000,
) -> int:
    """
    Reads an exams CSV stream and inserts rows into metrics.exam_records.
    Skips rows where patient_id or collection_date cannot be resolved.
    Returns the number of rows inserted.
    """
    total = 0
    mapping: Optional[dict] = None
    batch: list[dict] = []

    for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=chunk_size,
                              encoding="utf-8", on_bad_lines="warn"):
        if mapping is None:
            mapping = _RESOLVER.resolve(list(chunk.columns))
            logger.info("exams_columns_resolved hospital=%s mapping=%s", hospital_id, mapping)

        for _, row in chunk.iterrows():
            record = _build_record(row, mapping)
            if record is None:
                continue
            batch.append(record)
            if len(batch) >= _BATCH_SIZE:
                _flush(batch, db)
                total += len(batch)
                batch = []

    if batch:
        _flush(batch, db)
        total += len(batch)

    logger.info("exams_loaded hospital=%s total=%d", hospital_id, total)
    return total


def _build_record(row, mapping: dict) -> Optional[dict]:
    patient_id = _get(row, mapping, "patient_id")
    if not patient_id:
        return None

    collection_date = parse_date(_get(row, mapping, "collection_date"))
    if not collection_date:
        return None

    analyte     = _get(row, mapping, "analyte") or _get(row, mapping, "exam_group") or "UNKNOWN"
    exam_group  = _get(row, mapping, "exam_group")
    result_text = _get(row, mapping, "result_text")
    origin      = normalize_optional(_get(row, mapping, "origin"))

    # Numeric value: prefer pre-extracted column, fall back to text parsing
    result_num_raw = _get(row, mapping, "result_num")
    if result_num_raw:
        try:
            value = float(result_num_raw.replace(",", "."))
        except (ValueError, TypeError):
            value = extract_numeric(result_text or "", analyte)
    else:
        value = extract_numeric(result_text or "", analyte)

    if value is None:
        return None

    ref_low, ref_high = parse_reference_range(_get(row, mapping, "reference_range") or "")

    return {
        "patient_id":    patient_id.strip(),
        "analyte":       analyte.strip(),
        "date":          collection_date.isoformat(),
        "value":         value,
        "phase":         infer_phase(origin),
        "origin":        origin,
        "exam_group":    exam_group.strip() if exam_group else None,
        "value_text":    result_text.strip() if result_text else None,
        "unit":          normalize_optional(_get(row, mapping, "unit")),
        "ref_low":       ref_low,
        "ref_high":      ref_high,
        "attendance_id": _get(row, mapping, "attendance_id"),
    }


def _flush(batch: list[dict], db: PatientDB) -> None:
    by_patient: dict[str, list] = {}
    for row in batch:
        by_patient.setdefault(row["patient_id"], []).append(row)
    for patient_id, exams in by_patient.items():
        db.add_exams(patient_id, exams)


def _get(row, mapping: dict, concept: str) -> Optional[str]:
    col = mapping.get(concept)
    if col is None:
        return None
    val = row.get(col)
    if val is None or (isinstance(val, float)):
        return None
    return str(val).strip() or None
