"""
patients_adapter.py
Loads a FAPESP patients CSV into clinical.patients.

Expected source columns (resolved semantically, case-insensitive):
  patient_id, sex, birth_year, state_code, hospital_id
"""
import logging
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from integration.column_resolver import CLINICAL_SEMANTIC_MAP, ColumnResolver
from integration.fapesp.transforms import birth_year_to_age, normalize_optional, parse_birth_year
from infrastructure.mosaicfl_api.db import PatientDB

logger = logging.getLogger(__name__)

_REQUIRED = {"patient_id", "sex", "birth_year"}
_RESOLVER = ColumnResolver(CLINICAL_SEMANTIC_MAP, required=_REQUIRED)


def load_patients(
    file: TextIOWrapper,
    db: PatientDB,
    hospital_id: str,
    chunk_size: int = 10_000,
) -> int:
    """
    Reads a patients CSV stream and upserts rows into clinical.patients.
    Returns the number of rows processed.
    """
    total = 0
    first_chunk = True

    for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=chunk_size,
                              encoding="utf-8", on_bad_lines="warn"):
        if first_chunk:
            mapping = _RESOLVER.resolve(list(chunk.columns))
            logger.info("patients_columns_resolved hospital=%s mapping=%s", hospital_id, mapping)
            first_chunk = False

        for _, row in chunk.iterrows():
            patient_id = _get(row, mapping, "patient_id")
            if not patient_id:
                continue

            sex          = _get(row, mapping, "sex") or "U"
            birth_year   = parse_birth_year(_get(row, mapping, "birth_year"))
            state_code   = normalize_optional(_get(row, mapping, "state_code"))
            src_hosp     = normalize_optional(_get(row, mapping, "hospital_id")) or hospital_id
            municipality = normalize_optional(_get(row, mapping, "municipality"))
            cep_raw      = normalize_optional(_get(row, mapping, "cep_prefix"))
            cep_prefix   = cep_raw[:5] if cep_raw else None

            db.upsert_patient(
                patient_id   = patient_id.strip(),
                sex          = sex.strip().upper()[:1],
                age          = birth_year_to_age(birth_year),
                birth_year   = birth_year,
                state_code   = state_code,
                hospital_id  = src_hosp,
                municipality = municipality,
                cep_prefix   = cep_prefix,
            )
            total += 1

    logger.info("patients_loaded hospital=%s total=%d", hospital_id, total)
    return total


def _get(row, mapping: dict, concept: str) -> Optional[str]:
    col = mapping.get(concept)
    if col is None:
        return None
    val = row.get(col)
    if val is None or (isinstance(val, float)):
        return None
    return str(val).strip() or None
