"""lookups.py — Cache de referências canônicas e aliases, resolução e classificação de valores.

Chamados uma vez por job (não por linha) — os resultados são reutilizados em todo o chunk.
"""
from typing import Optional


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
