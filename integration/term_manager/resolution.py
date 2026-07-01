"""resolution.py — Resolução de nome bruto → canonical via cache de aliases ativos."""
from __future__ import annotations

from typing import TYPE_CHECKING

from integration.column_resolver import normalize

if TYPE_CHECKING:
    from sqlalchemy import Connection


def _to_canonical(raw_name: str) -> str:
    """Converte nome bruto para a forma canônica: MAIUSCULAS_COM_UNDERSCORE."""
    return normalize(raw_name).upper()


def _load_alias_cache(conn: "Connection", term_type: str) -> dict[str, str]:
    """Retorna {normalized_key: canonical} para entradas active=TRUE.

    Indexa tanto os aliases quanto os próprios canônicos normalizados, para
    que 'Eritrócitos' encontre 'ERITROCITOS' mesmo que só exista o alias
    'Eritrocitos' — ou nenhum alias, apenas o canonical em si.

    normalize("Eritrócitos") == normalize("Eritrocitos") == normalize("ERITROCITOS")
    → todos resolvem para "ERITROCITOS".
    """
    from sqlalchemy import text

    rows = conn.execute(
        text("""
            SELECT canonical, alias
            FROM knowledge.term_dictionary
            WHERE term_type = :tt AND active = TRUE
        """),
        {"tt": term_type},
    ).fetchall()

    cache: dict[str, str] = {}
    for canonical, alias in rows:
        cache[normalize(alias)]     = canonical  # alias normalizado → canonical
        cache[normalize(canonical)] = canonical  # canonical normalizado → canonical
    return cache


def _resolve_one(raw_name: str, cache: dict[str, str]) -> str | None:
    """Tenta resolver o canonical a partir do cache. Retorna None se não encontrado.

    Usa apenas exact match após normalize() — starts-with é intencionalmente omitido
    para analitos porque prefixo compartilhado não implica equivalência clínica
    (ex: CREATININA ≠ CREATININA_URINA). Aliases verdadeiros devem ser cadastrados
    explicitamente em term_dictionary.
    """
    norm = normalize(raw_name)
    return cache.get(norm)
