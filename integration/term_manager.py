"""
term_manager.py
Gerencia o ciclo de vida dos termos em knowledge.term_dictionary.

Fluxo operacional:
  1. validate_analytes_before_load() — roda ANTES de qualquer carga de dados.
     Auto-registra analitos desconhecidos com active=FALSE e bloqueia a carga
     se houver pendências.
  2. list_pending_terms()            — lista o que está pendente de revisão.
  3. correct_term()                  — corrige o canonical proposto (ex: WBC → LEUCOCITOS).
  4. activate_term()                 — ativa um termo após confirmação de que o canonical está correto.
  5. activate_all_auto_normalized()  — ativação em lote para variantes de grafia sem aliases verdadeiros.

Convenção de canonização:
  normalize(raw_name).upper() — letras maiúsculas, underscores no lugar de
  não-alfanuméricos, sem acentos.
  Exemplos: "Leucócitos" → "LEUCOCITOS"
            "Proteína C-Reativa" → "PROTEINA_C_REATIVA"
            "D-Dímero" → "D_DIMERO"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from integration.column_resolver import normalize

if TYPE_CHECKING:
    from sqlalchemy import Connection

logger = logging.getLogger(__name__)

_TERM_TYPE_ANALYTE = "analyte"
_SOURCE_AUTO = "AUTO_NORMALIZED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PendingTerm:
    alias: str
    canonical_proposto: str
    source: str
    term_type: str

    def __str__(self) -> str:
        return (
            f"  alias={self.alias!r:45s}  "
            f"canonical_proposto={self.canonical_proposto!r:30s}  "
            f"source={self.source!r}"
        )


@dataclass
class ValidationResult:
    total_analitos: int
    resolvidos: int
    pendentes: list[PendingTerm] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.pendentes) == 0

    def print_report(self) -> None:
        print(f"\nValidação de analitos: {self.resolvidos}/{self.total_analitos} resolvidos")
        if self.pendentes:
            print(f"\n{'─'*100}")
            print(f"{'ALIAS ORIGINAL':<45}  {'CANONICAL PROPOSTO':<30}  FONTE")
            print(f"{'─'*100}")
            for p in sorted(self.pendentes, key=lambda x: x.canonical_proposto):
                print(p)
            print(f"{'─'*100}")
            print(
                f"\n⚠ {len(self.pendentes)} termo(s) com active=FALSE. "
                "Corrija ou ative antes de prosseguir com a carga.\n"
                "  → list_pending_terms()           para ver todos os pendentes\n"
                "  → activate_term(alias, ...)      para ativar o canonical proposto\n"
                "  → correct_term(alias, novo, ...) para corrigir o canonical e ativar\n"
            )
        else:
            print("✓ Todos os analitos têm canonical ativo. Carga liberada.\n")


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------

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


def _register_pending(
    raw_name: str,
    proposed_canonical: str,
    term_type: str,
    source: str,
    conn: "Connection",
) -> None:
    """Insere o termo com active=FALSE se ainda não existir."""
    from sqlalchemy import text

    conn.execute(
        text("""
            INSERT INTO knowledge.term_dictionary
                (term_type, canonical, alias, source, active)
            VALUES (:tt, :canonical, :alias, :source, FALSE)
            ON CONFLICT (term_type, canonical, alias) DO NOTHING
        """),
        {
            "tt": term_type,
            "canonical": proposed_canonical,
            "alias": raw_name,
            "source": source,
        },
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def validate_analytes_before_load(
    source_analytes: list[str],
    conn: "Connection",
    source: str = _SOURCE_AUTO,
) -> ValidationResult:
    """Valida se todos os analitos da fonte têm canonical ativo em term_dictionary.

    Para cada analito sem mapeamento ativo:
      - Propõe canonical = normalize(raw_name).upper()
      - Registra em term_dictionary com active=FALSE
      - Inclui no relatório de pendências

    A carga só deve prosseguir se result.ok == True.
    """
    cache = _load_alias_cache(conn, _TERM_TYPE_ANALYTE)
    unique = list(dict.fromkeys(source_analytes))  # preserva ordem, remove duplicatas

    resolvidos = 0
    pendentes: list[PendingTerm] = []

    for raw in unique:
        canonical = _resolve_one(raw, cache)
        if canonical:
            resolvidos += 1
        else:
            proposed = _to_canonical(raw)
            _register_pending(raw, proposed, _TERM_TYPE_ANALYTE, source, conn)
            pendentes.append(PendingTerm(
                alias=raw,
                canonical_proposto=proposed,
                source=source,
                term_type=_TERM_TYPE_ANALYTE,
            ))

    conn.commit()
    result = ValidationResult(
        total_analitos=len(unique),
        resolvidos=resolvidos,
        pendentes=pendentes,
    )
    result.print_report()
    return result


def list_pending_terms(
    conn: "Connection",
    term_type: str = _TERM_TYPE_ANALYTE,
) -> list[PendingTerm]:
    """Lista todos os termos com active=FALSE aguardando revisão do operador."""
    from sqlalchemy import text

    rows = conn.execute(
        text("""
            SELECT alias, canonical, source
            FROM knowledge.term_dictionary
            WHERE term_type = :tt AND active = FALSE
            ORDER BY canonical, alias
        """),
        {"tt": term_type},
    ).fetchall()

    pending = [
        PendingTerm(alias=r.alias, canonical_proposto=r.canonical, source=r.source, term_type=term_type)
        for r in rows
    ]

    if not pending:
        print(f"Nenhum termo pendente para term_type='{term_type}'.")
    else:
        print(f"\n{'─'*100}")
        print(f"{'ALIAS ORIGINAL':<45}  {'CANONICAL PROPOSTO':<30}  FONTE")
        print(f"{'─'*100}")
        for p in pending:
            print(p)
        print(f"{'─'*100}")
        print(f"\nTotal: {len(pending)} pendente(s).\n")

    return pending


def activate_term(
    alias: str,
    conn: "Connection",
    term_type: str = _TERM_TYPE_ANALYTE,
) -> bool:
    """Ativa o canonical proposto automaticamente para um alias.

    Use quando o canonical proposto pela auto-normalização está correto
    (ex: "Eritrócitos" → "ERITROCITOS" ✓).
    Retorna True se ativado, False se o alias não foi encontrado como pendente.
    """
    from sqlalchemy import text

    result = conn.execute(
        text("""
            UPDATE knowledge.term_dictionary
               SET active = TRUE
             WHERE term_type = :tt
               AND alias     = :alias
               AND active    = FALSE
        """),
        {"tt": term_type, "alias": alias},
    )
    conn.commit()

    activated = result.rowcount > 0
    if activated:
        row = conn.execute(
            text("SELECT canonical FROM knowledge.term_dictionary WHERE term_type=:tt AND alias=:alias"),
            {"tt": term_type, "alias": alias},
        ).fetchone()
        canonical = row.canonical if row else "?"
        print(f"✓ Ativado: {alias!r} → {canonical!r}")
    else:
        print(f"✗ Não encontrado como pendente: alias={alias!r}, term_type={term_type!r}")

    return activated


def correct_term(
    alias: str,
    correct_canonical: str,
    conn: "Connection",
    term_type: str = _TERM_TYPE_ANALYTE,
) -> bool:
    """Corrige o canonical de um termo pendente e o ativa.

    Use quando o canonical proposto está ERRADO e precisa ser corrigido
    antes de ativar (ex: "WBC" foi proposto como "WBC" mas deve ser "LEUCOCITOS").

    O correct_canonical deve seguir a convenção: MAIUSCULAS_COM_UNDERSCORE.
    Retorna True se corrigido e ativado, False se o alias não foi encontrado.
    """
    from sqlalchemy import text

    canonical_norm = correct_canonical.strip().upper()

    result = conn.execute(
        text("""
            UPDATE knowledge.term_dictionary
               SET canonical = :canonical,
                   active    = TRUE
             WHERE term_type = :tt
               AND alias     = :alias
        """),
        {"canonical": canonical_norm, "tt": term_type, "alias": alias},
    )
    conn.commit()

    corrected = result.rowcount > 0
    if corrected:
        print(f"✓ Corrigido e ativado: {alias!r} → {canonical_norm!r}")
    else:
        print(f"✗ Não encontrado: alias={alias!r}, term_type={term_type!r}")

    return corrected


def activate_all_auto_normalized(
    conn: "Connection",
    term_type: str = _TERM_TYPE_ANALYTE,
) -> int:
    """Ativa em lote todos os termos auto-normalizados (source=AUTO_NORMALIZED).

    Seguro para variantes de grafia (acentos, capitalização) onde o canonical
    proposto pela função normalize().upper() já é correto.
    NÃO use se houver aliases verdadeiros (WBC, CRP) pendentes — corrija-os
    individualmente com correct_term() antes.

    Retorna o número de termos ativados.
    """
    from sqlalchemy import text

    from sqlalchemy import text

    # Canônicos já ativos — usados para detectar colisões de nome
    active_canonicals: set[str] = {
        normalize(r.canonical)
        for r in conn.execute(
            text("SELECT DISTINCT canonical FROM knowledge.term_dictionary WHERE term_type=:tt AND active=TRUE"),
            {"tt": term_type},
        ).fetchall()
    }

    pending = list_pending_terms(conn, term_type)

    # Alias verdadeiro: normalize(alias).upper() difere do canonical proposto
    aliases_verdadeiros = [
        p for p in pending
        if normalize(p.alias).upper() != p.canonical_proposto
    ]

    # Colisão: o canonical proposto é diferente de um canônico ativo existente
    # mas normaliza para o mesmo valor (ex: BILIRRUBINA_TOTAL vs BILIRRUBINA_TOT)
    colisoes = [
        p for p in pending
        if p not in aliases_verdadeiros
        and any(
            normalize(p.canonical_proposto) == ac and p.canonical_proposto.upper() != ac.upper()
            for ac in active_canonicals
        )
    ]

    problemas = aliases_verdadeiros + colisoes
    if problemas:
        if aliases_verdadeiros:
            print(f"\n⚠ {len(aliases_verdadeiros)} alias(es) verdadeiro(s) — canonical proposto pode estar errado:\n")
            for p in aliases_verdadeiros:
                print(f"  {p}")
        if colisoes:
            print(f"\n⚠ {len(colisoes)} colisão(ões) de nome — canonical proposto duplica um já existente:\n")
            for p in colisoes:
                # Encontra o canônico ativo que colide
                colide_com = next(
                    ac for ac in active_canonicals
                    if normalize(p.canonical_proposto) == ac
                )
                print(f"  {p}  ← colide com '{colide_com.upper()}'")
        print(
            "\nUse correct_term(alias, canonical_correto, conn) para cada um acima "
            "e então chame activate_all_auto_normalized() novamente.\n"
        )
        return 0

    result = conn.execute(
        text("""
            UPDATE knowledge.term_dictionary
               SET active = TRUE
             WHERE term_type = :tt
               AND source    = :src
               AND active    = FALSE
        """),
        {"tt": term_type, "src": _SOURCE_AUTO},
    )
    conn.commit()

    count = result.rowcount
    print(f"✓ {count} termo(s) auto-normalizados ativados.")
    return count
