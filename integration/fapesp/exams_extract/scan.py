"""scan.py — Fase 1: escaneia o CSV e valida analitos contra term_dictionary, sem inserir dados."""
import logging
from io import TextIOWrapper
from typing import Optional

import pandas as pd

from integration.term_manager import ValidationResult, validate_analytes_before_load
from infrastructure.mosaicfl_api.db import PatientDB

from .column_mapping import _RESOLVER

logger = logging.getLogger(__name__)


def scan_analytes(
    file: TextIOWrapper,
    db: PatientDB,
    hospital_id: str,
    source: str,
    chunk_size: int = 50_000,
) -> ValidationResult:
    """Streams the CSV, collects all analyte names, validates against term_dictionary.

    Does NOT insert any data. Run before load_exams and let the operator review
    pending terms with term_manager before proceeding to load.

    Returns a ValidationResult — result.ok == True means load_exams can proceed.
    """
    mapping: Optional[dict] = None
    all_analytes: set[str] = set()

    for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=chunk_size,
                             encoding="utf-8", on_bad_lines="warn"):
        if mapping is None:
            mapping = _RESOLVER.resolve(list(chunk.columns))
            logger.info("scan_analytes hospital=%s columns_resolved=%s", hospital_id, mapping)

        if mapping and "analyte" in mapping:
            all_analytes.update(
                chunk[mapping["analyte"]].dropna().astype(str).str.strip().unique()
            )

    logger.info("scan_analytes hospital=%s distinct_analytes=%d", hospital_id, len(all_analytes))

    with db._engine.connect() as conn:
        result = validate_analytes_before_load(all_analytes, conn, source=source)

    return result
