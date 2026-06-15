"""
exams_extract.py
Loads an exams CSV stream into metrics.exam_records.

Canonical normalization happens at ingestion:
  1. Raw analyte name is resolved to its canonical form via knowledge.term_dictionary.
  2. The value is classified (HIGH/NORMAL/LOW/NO_REF) using knowledge.analyte_references.
  3. canonical_ref_low, canonical_ref_high, and classification are stored in exam_records.

Two-phase workflow (run in order):
  1. scan_analytes() — streams the CSV, collects all analyte names, validates against
     term_dictionary. Registers unknowns as inactive. Does NOT insert data.
     Run this first and let the operator review pending terms before loading.
  2. load_exams() — streams the CSV chunk by chunk, resolves canonical names,
     classifies values, inserts records. Does not validate or block.

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
from integration.term_manager import validate_analytes_before_load, ValidationResult
from infrastructure.mosaicfl_api.db import PatientDB

logger = logging.getLogger(__name__)

_REQUIRED = {"patient_id", "collection_date", "analyte", "result_text"}
_RESOLVER = ColumnResolver(CLINICAL_SEMANTIC_MAP, required=_REQUIRED)

_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Lookup helpers — called once per job, not per row
# ---------------------------------------------------------------------------

def _load_canonical_refs(engine) -> dict[str, tuple[float, float]]:
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT canonical, ref_low, ref_high
            FROM knowledge.analyte_references
            WHERE sex IS NULL
        """)).fetchall()
    return {r.canonical: (float(r.ref_low), float(r.ref_high)) for r in rows}


def _load_alias_cache(engine) -> dict[str, str]:
    from sqlalchemy import text
    from integration.column_resolver import normalize
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT canonical, alias
            FROM knowledge.term_dictionary
            WHERE term_type = 'analyte' AND active = TRUE
        """)).fetchall()
    cache: dict[str, str] = {}
    for canonical, alias in rows:
        cache[normalize(alias)]     = canonical
        cache[normalize(canonical)] = canonical
    return cache


def _resolve_canonical(raw_name: str, alias_cache: dict[str, str]) -> str:
    from integration.column_resolver import normalize
    norm = normalize(raw_name)
    if norm in alias_cache:
        return alias_cache[norm]
    # Alias not in term_dictionary — auto-normalize as fallback.
    # scan_analytes will have registered this as pending; operator reviews before load.
    return norm.upper()


def _classify(value: float, canonical: str, canonical_refs: dict) -> Optional[str]:
    if not canonical_refs:
        return None
    if canonical not in canonical_refs:
        return "NO_REF"
    ref_low, ref_high = canonical_refs[canonical]
    if ref_low == 0.0 and ref_high == 0.0:
        return "NO_REF"
    if value < ref_low:
        return "LOW"
    if value > ref_high:
        return "HIGH"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Phase 1 — scan without loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 2 — load
# ---------------------------------------------------------------------------

def load_exams(
    file: TextIOWrapper,
    db: PatientDB,
    hospital_id: str,
    source: str,
    chunk_size: int = 50_000,
) -> int:
    """Streams the exams CSV and inserts rows into metrics.exam_records.

    Resolves canonical analyte names and classifies values at ingestion time.
    Does not validate or block — run scan_analytes first.

    Returns the number of rows inserted.
    """
    engine = db._engine

    alias_cache    = _load_alias_cache(engine)
    canonical_refs = _load_canonical_refs(engine)

    if not alias_cache:
        logger.warning(
            "load_exams hospital=%s alias_cache_empty — "
            "analytes will be stored as auto-normalized names; "
            "run scan_analytes first to populate term_dictionary",
            hospital_id,
        )
    if not canonical_refs:
        logger.warning(
            "load_exams hospital=%s canonical_refs_empty — "
            "classification will be NULL; "
            "run compute_analyte_references.py + backfill after loading",
            hospital_id,
        )

    total:   int           = 0
    mapping: Optional[dict] = None
    batch:   list[dict]    = []

    for chunk in pd.read_csv(file, sep="|", dtype=str, chunksize=chunk_size,
                             encoding="utf-8", on_bad_lines="warn"):
        if mapping is None:
            mapping = _RESOLVER.resolve(list(chunk.columns))
            logger.info("load_exams hospital=%s columns_resolved=%s", hospital_id, mapping)

        records = _process_chunk(chunk, mapping, alias_cache, canonical_refs)
        batch.extend(records)

        if len(batch) >= _BATCH_SIZE:
            _flush(batch, db)
            total += len(batch)
            batch = []

    if batch:
        _flush(batch, db)
        total += len(batch)

    logger.info("load_exams hospital=%s source=%s total=%d", hospital_id, source, total)
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_chunk(
    chunk: pd.DataFrame,
    mapping: dict,
    alias_cache: dict[str, str],
    canonical_refs: dict[str, tuple[float, float]],
) -> list[dict]:
    """Vectorized chunk processing — substitui iterrows.

    Opera colunas inteiras via pandas/numpy. Usa apply() só onde o transform
    é complexo (parse_date, parse_reference_range). Filtra linhas inválidas
    antes das operações caras, reduzindo o trabalho sobre o subset válido.
    """
    def _col(concept: str) -> pd.Series:
        c = mapping.get(concept)
        return chunk[c] if c and c in chunk.columns else pd.Series(
            [None] * len(chunk), index=chunk.index, dtype=object
        )

    # 1. patient_id — obrigatório
    pid = _col("patient_id").astype(str).str.strip()
    valid = pid.notna() & ~pid.isin(["", "nan", "None", "NAN"])

    # 2. collection_date — obrigatório; apply é 10-20× mais rápido que iterrows
    parsed_date = _col("collection_date").where(valid).apply(
        lambda v: parse_date(str(v)) if pd.notna(v) else None
    )
    valid &= parsed_date.notna()

    # 3. Analyte (preferência: analyte; fallback: exam_group)
    ana_s = _col("analyte").astype(str).str.strip()
    grp_s = _col("exam_group").astype(str).str.strip()
    raw_analyte = ana_s.where(ana_s.notna() & ana_s.ne("nan") & ana_s.ne(""), grp_s)
    raw_analyte = raw_analyte.fillna("UNKNOWN")
    raw_analyte = raw_analyte.where(raw_analyte.ne("") & raw_analyte.ne("nan"), "UNKNOWN")

    # 4. Valor numérico — obrigatório
    result_text = _col("result_text").fillna("").astype(str)
    result_num_raw = _col("result_num").fillna("").astype(str)

    def _parse_value(num_s: str, txt: str, analyte: str) -> Optional[float]:
        num_s = num_s.strip()
        if num_s and num_s not in ("nan", "None", ""):
            try:
                return float(num_s.replace(",", "."))
            except (ValueError, TypeError):
                pass
        return extract_numeric(txt, analyte)

    value = pd.Series(
        [_parse_value(n, t, a) for n, t, a in zip(result_num_raw, result_text, raw_analyte)],
        index=chunk.index,
    )
    valid &= value.notna()

    if not valid.any():
        return []

    # Aplica máscara ao subset válido — todas as operações seguintes são no subset
    pid         = pid[valid]
    parsed_date = parsed_date[valid]
    raw_analyte = raw_analyte[valid]
    result_text = result_text[valid]
    value       = value[valid]

    # 5. Reference range (apply no subset filtrado)
    ref_parsed  = _col("reference_range")[valid].fillna("").astype(str).apply(parse_reference_range)
    ref_low     = ref_parsed.apply(lambda t: t[0])
    ref_high    = ref_parsed.apply(lambda t: t[1])

    # 6. Colunas opcionais
    origin   = _col("origin")[valid].apply(
        lambda v: normalize_optional(str(v)) if pd.notna(v) else None
    )
    unit     = _col("unit")[valid].apply(
        lambda v: normalize_optional(str(v)) if pd.notna(v) else None
    )
    exam_grp = _col("exam_group")[valid]
    att_id   = _col("attendance_id")[valid]

    # 7. Resolução canônica (dict lookup via apply — vectorizado)
    canonical = raw_analyte.apply(lambda n: _resolve_canonical(n, alias_cache))

    # 8. Classificação (apply no subset — evita re-criação de objetos por linha)
    def _cls(val: float, can: str):
        if not canonical_refs:
            return None, None, None
        pair = canonical_refs.get(can)
        if pair is None:
            return "NO_REF", None, None
        lo, hi = pair
        if lo == 0.0 and hi == 0.0:
            return "NO_REF", lo, hi
        if val < lo:
            return "LOW", lo, hi
        if val > hi:
            return "HIGH", lo, hi
        return "NORMAL", lo, hi

    classified  = pd.Series(
        [_cls(v, c) for v, c in zip(value, canonical)],
        index=value.index,
    )
    classif    = classified.apply(lambda t: t[0])
    cref_low   = classified.apply(lambda t: t[1])
    cref_high  = classified.apply(lambda t: t[2])

    # 9. Phase (map — pura lookup de dict, sem apply)
    phase = origin.apply(infer_phase)

    # 10. Construção das dicts (iteração apenas sobre linhas válidas)
    records: list[dict] = []
    for p, d, rt, v, rl, rh, ori, u, eg, ai, can, cls, crl, crh, ph in zip(
        pid, parsed_date, result_text, value,
        ref_low, ref_high, origin, unit, exam_grp, att_id,
        canonical, classif, cref_low, cref_high, phase,
    ):
        records.append({
            "patient_id":         p,
            "analyte":            can,
            "date":               d.isoformat(),
            "value":              v,
            "phase":              ph,
            "origin":             ori,
            "exam_group":         str(eg).strip() if pd.notna(eg) and str(eg) not in ("nan", "None") else None,
            "value_text":         rt.strip() if rt else None,
            "unit":               u,
            "ref_low":            rl,
            "ref_high":           rh,
            "attendance_id":      str(ai).strip() if pd.notna(ai) and str(ai) not in ("nan", "None") else None,
            "canonical_ref_low":  crl,
            "canonical_ref_high": crh,
            "classification":     cls,
        })
    return records


def _flush(batch: list[dict], db: PatientDB) -> None:
    """Insere o batch em uma única transação — sem agrupamento por paciente."""
    db.add_exams_bulk(batch)
