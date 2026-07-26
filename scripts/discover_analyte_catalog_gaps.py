"""
discover_analyte_catalog_gaps.py
Descobre analitos presentes em metrics.exam_records sem canonical ativo em
knowledge.term_dictionary e registra os que atendem a um critério mínimo de
relevância — fecha a lacuna que hoje limita build_standard_vocab.py a um
catálogo de 15 analitos curados manualmente (painel de gravidade COVID),
enquanto os dados reais têm centenas de analitos distintos (achado 2026-07-26,
ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md).

Sem estar no catálogo ativo, um analito não ganha token no vocabulário
federado (build_standard_vocab.py) e vira <UNK> na tokenização
(SequencePipeline._build_tensors) — não é excluído, vira ruído genérico
ocupando posição na sequência.

NÃO tenta unificar sinônimos com canônicos já existentes (ex: 'ALT_TGP' vs
'TGP' — ver migration 022_fix_dead_curated_canonicals para esse caso
específico, já corrigido). Cada string já normalizada em exam_records.analyte
vira seu próprio canonical (canonical == alias); unificação de sinônimos é
decisão clínica que exige revisão humana — fora do escopo deste script.
warn_possible_synonyms() é só um aviso heurístico no --dry-run, não bloqueia.

Pré-requisito: rode scripts/compute_analyte_references.py (já com a correção
de insert_no_ref_placeholders) ANTES deste script — sem isso, boa parte dos
candidatos de alto volume aparece sem classification válida e o critério de
"referência real" abaixo os rejeita incorretamente.

Uso:
    python3 scripts/discover_analyte_catalog_gaps.py --dry-run
    python3 scripts/discover_analyte_catalog_gaps.py
    python3 scripts/discover_analyte_catalog_gaps.py --min-records 50 --min-hospitals 2
    python3 scripts/discover_analyte_catalog_gaps.py --include-no-ref
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def find_candidates(conn, min_records: int = 100, min_hospitals: int = 1) -> list[dict]:
    """Analitos fora do catálogo ativo, com volume mínimo de registros/hospitais.

    has_real_ref distingue quem já tem referência institucional de verdade
    (source='MEDIA_HOSPITAIS_PARTICIPANTES', ver compute_analyte_references.py)
    de quem só tem o placeholder NO_REF (source='SEM_REFERENCIA_INSTITUCIONAL')
    ou nenhuma referência ainda — usado por select_insertable() pra decidir o
    tier de ativação.
    """
    from sqlalchemy import text

    rows = conn.execute(text("""
        WITH per_hospital AS (
            SELECT a.hospital_id, e.analyte, COUNT(*) AS n
            FROM metrics.exam_records e
            JOIN clinical.attendances a ON a.attendance_id = e.attendance_id
            WHERE e.analyte IS NOT NULL
            GROUP BY a.hospital_id, e.analyte
        ),
        aggregated AS (
            SELECT analyte, COUNT(DISTINCT hospital_id) AS n_hospitals, SUM(n) AS n_records
            FROM per_hospital
            GROUP BY analyte
        )
        SELECT ag.analyte, ag.n_hospitals, ag.n_records,
               ar.ref_low, ar.ref_high, ar.source AS ref_source
        FROM aggregated ag
        LEFT JOIN knowledge.analyte_references ar
               ON ar.canonical = ag.analyte AND ar.sex IS NULL
        LEFT JOIN knowledge.term_dictionary td
               ON td.term_type = 'analyte' AND td.canonical = ag.analyte AND td.active = TRUE
        WHERE td.canonical IS NULL
          AND ag.n_records   >= :min_records
          AND ag.n_hospitals >= :min_hospitals
        ORDER BY ag.n_records DESC
    """), {"min_records": min_records, "min_hospitals": min_hospitals}).fetchall()

    candidates = []
    for r in rows:
        has_real_ref = (
            r.ref_source == "MEDIA_HOSPITAIS_PARTICIPANTES"
            and r.ref_low is not None
            and not (float(r.ref_low) == 0.0 and float(r.ref_high) == 0.0)
        )
        candidates.append({
            "analyte":     r.analyte,
            "n_hospitals": int(r.n_hospitals),
            "n_records":   int(r.n_records),
            "has_real_ref": has_real_ref,
            "ref_source":  r.ref_source,
        })
    return candidates


def select_insertable(candidates: list[dict], include_no_ref: bool = False) -> list[dict]:
    """Critério de ativação: tier padrão exige referência institucional real
    (garante que o catálogo só cresce com analitos que carregam sinal
    HIGH/NORMAL/LOW, não só NO_REF). --include-no-ref amplia pro tier
    opcional — ainda melhor que <UNK>, mas sem sinal de faixa."""
    if include_no_ref:
        return list(candidates)
    return [c for c in candidates if c["has_real_ref"]]


def warn_possible_synonyms(candidates: list[dict], active_canonicals: set[str]) -> list[str]:
    """Heurística best-effort (substring) — só para aviso no relatório --dry-run,
    nunca bloqueia inserção. Documentadamente imperfeita: pega 'ALT_TGP' vs
    'TGP', mas NÃO pega 'DIMEROS_D_QUANTITATIVO' vs 'D_DIMERO' (sem substring
    em comum) — esse tipo de caso precisa de revisão manual, ver migration
    022_fix_dead_curated_canonicals para um exemplo já corrigido dessa forma."""
    warnings = []
    for c in candidates:
        for ac in active_canonicals:
            if ac in c["analyte"] or c["analyte"] in ac:
                warnings.append(f"  aviso: {c['analyte']!r} pode ser sinônimo de canonical ativo {ac!r}")
    return warnings


def get_active_canonicals(conn) -> set[str]:
    from sqlalchemy import text
    rows = conn.execute(text("""
        SELECT DISTINCT canonical FROM knowledge.term_dictionary
        WHERE term_type = 'analyte' AND active = TRUE
    """)).fetchall()
    return {r.canonical for r in rows}


def insert_candidates(conn, candidates: list[dict]) -> int:
    """canonical == alias: exam_records.analyte já vem normalizado (confirmado
    ao vivo — LINFOCITOS, HEMOGLOBINA_CORPUSCULAR_MEDIA etc., sem alias bruto
    de hospital pra preservar). source='AUTO_DISCOVERED' distingue de entradas
    curadas manualmente (source='FAPESP'/'CLINICAL_PATH'/'MANUAL', ver 008).

    DO UPDATE SET active = TRUE (não DO NOTHING): reativa um canonical que já
    existia na chave natural (term_type, canonical, alias) mas foi desativado
    (active=FALSE) — sem isso, um analito reativado nunca voltaria a aparecer
    em build_standard_vocab() (filtra WHERE active = TRUE), mesmo com o INSERT
    "funcionando" sem erro. Achado 2026-07-26 discutindo a validação do
    vocabulário federado bidirecional (ver strategy/core.py::
    _discover_and_curate_vocab, único chamador em produção deste código)."""
    from sqlalchemy import text
    for c in candidates:
        conn.execute(text("""
            INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source, active)
            VALUES ('analyte', :canonical, :canonical, 'AUTO_DISCOVERED', TRUE)
            ON CONFLICT (term_type, canonical, alias) DO UPDATE SET active = TRUE
        """), {"canonical": c["analyte"]})
    conn.commit()
    return len(candidates)


def print_report(candidates: list[dict], selected: list[dict], warnings: list[str]) -> None:
    print(f"\n{'Analito':<45} {'N hosp':>7} {'N registros':>12}  {'Ref. real':>9}")
    print("─" * 78)
    selected_analytes = {c["analyte"] for c in selected}
    for c in candidates:
        marker = "✓" if c["analyte"] in selected_analytes else " "
        print(
            f"{marker} {c['analyte']:<43} {c['n_hospitals']:>7} {c['n_records']:>12,}  "
            f"{'sim' if c['has_real_ref'] else 'não':>9}"
        )
    print(f"\nCandidatos encontrados: {len(candidates)} | Selecionados pra ativação: {len(selected)}\n")
    if warnings:
        print("Possíveis sinônimos de canônicos já ativos (revisar manualmente):")
        for w in warnings:
            print(w)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Mostra candidatos, não grava")
    parser.add_argument("--min-records", type=int, default=100,
                        help="Mínimo de registros pra considerar o analito (default: 100)")
    parser.add_argument("--min-hospitals", type=int, default=1,
                        help="Mínimo de hospitais distintos (default: 1)")
    parser.add_argument("--include-no-ref", action="store_true",
                        help="Também ativa analitos sem referência institucional real "
                             "(só o placeholder NO_REF) — tier opcional, desligado por padrão")
    args = parser.parse_args()

    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        log.error("FL_DB_URL não definida.")
        sys.exit(1)

    from sqlalchemy import create_engine
    engine = create_engine(db_url)

    with engine.connect() as conn:
        log.info("Procurando analitos fora do catálogo ativo ...")
        candidates = find_candidates(conn, min_records=args.min_records, min_hospitals=args.min_hospitals)
        active = get_active_canonicals(conn)
        selected = select_insertable(candidates, include_no_ref=args.include_no_ref)
        warnings = warn_possible_synonyms(selected, active)
        print_report(candidates, selected, warnings)

        if args.dry_run:
            log.info("Modo dry-run — nenhum dado gravado.")
            return

        n = insert_candidates(conn, selected)
        log.info(f"  {n} analitos ativados em knowledge.term_dictionary.")
        log.info(
            "  Rode scripts/build_standard_vocab.py em seguida pra reconstruir "
            "o vocabulário federado com o catálogo expandido."
        )


if __name__ == "__main__":
    main()
