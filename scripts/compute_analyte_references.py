"""
compute_analyte_references.py
Calcula e persiste as referências canônicas em knowledge.analyte_references.

Metodologia — média em dois níveis:
  1. Por hospital: AVG(ref_low) e AVG(ref_high) para cada (analito, hospital)
  2. Entre hospitais: AVG das médias anteriores para cada analito
  Cada hospital tem peso igual, independente do volume de registros.

Uso:
    python scripts/compute_analyte_references.py
    python scripts/compute_analyte_references.py --dry-run   # só exibe, não grava
    python scripts/compute_analyte_references.py --min-hospitals 2

Reexecute sempre que um novo hospital entrar na federação.
Após reexecutar, identifique registros com classificação desatualizada:
    SELECT COUNT(*) FROM metrics.exam_records e
    JOIN knowledge.analyte_references r ON r.canonical = e.analyte AND r.sex IS NULL
    WHERE e.canonical_ref_low  IS DISTINCT FROM r.ref_low
       OR e.canonical_ref_high IS DISTINCT FROM r.ref_high;
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

_CLASSIFICATION_VALUES = ("HIGH", "NORMAL", "LOW", "NO_REF")


def compute(conn, min_hospitals: int = 1) -> list[dict]:
    """Executa o cálculo em dois níveis e retorna as linhas a gravar."""
    from sqlalchemy import text

    rows = conn.execute(text("""
        WITH per_hospital AS (
            SELECT
                a.hospital_id,
                e.analyte                   AS canonical,
                COUNT(*)                    AS n_records,
                AVG(e.ref_low)              AS avg_ref_low,
                AVG(e.ref_high)             AS avg_ref_high
            FROM metrics.exam_records e
            JOIN clinical.attendances a ON a.attendance_id = e.attendance_id
            WHERE e.analyte   IS NOT NULL
              AND e.ref_low   IS NOT NULL
              AND e.ref_high  IS NOT NULL
              AND (e.ref_low > 0 OR e.ref_high > 0)
            GROUP BY a.hospital_id, e.analyte
        )
        SELECT
            canonical,
            COUNT(DISTINCT hospital_id)     AS n_hospitals,
            AVG(avg_ref_low)                AS ref_low,
            AVG(avg_ref_high)               AS ref_high
        FROM per_hospital
        GROUP BY canonical
        HAVING COUNT(DISTINCT hospital_id) >= :min_h
        ORDER BY canonical
    """), {"min_h": min_hospitals}).fetchall()

    return [
        {
            "canonical":   r.canonical,
            "sex":         None,
            "ref_low":     round(float(r.ref_low),  4),
            "ref_high":    round(float(r.ref_high), 4),
            "n_hospitals": int(r.n_hospitals),
            "source":      "MEDIA_HOSPITAIS_PARTICIPANTES",
        }
        for r in rows
    ]


def persist(conn, entries: list[dict]) -> int:
    """Grava as entradas em knowledge.analyte_references (upsert)."""
    from sqlalchemy import text

    upserted = 0
    for e in entries:
        conn.execute(text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:canonical, :sex, :ref_low, :ref_high, :n_hospitals, :source, NOW())
            ON CONFLICT (canonical, sex)
            DO UPDATE SET
                ref_low     = EXCLUDED.ref_low,
                ref_high    = EXCLUDED.ref_high,
                n_hospitals = EXCLUDED.n_hospitals,
                source      = EXCLUDED.source,
                computed_at = NOW()
        """), e)
        upserted += 1

    conn.commit()
    return upserted


def classify(value: float, ref_low: float | None, ref_high: float | None) -> str:
    """Classifica um valor em relação aos refs canônicos."""
    if ref_low is None or ref_high is None:
        return "NO_REF"
    if ref_low == 0.0 and ref_high == 0.0:
        return "NO_REF"
    if value < ref_low:
        return "LOW"
    if value > ref_high:
        return "HIGH"
    return "NORMAL"


def backfill_classifications(conn) -> int:
    """
    Preenche classification, canonical_ref_low e canonical_ref_high em dois casos:
      1. classification IS NULL  — registros carregados antes de analyte_references existir.
      2. classification = 'NO_REF' — registros cujo analito não tinha referência canônica
         na época da ingestão mas passou a ter após recomputação (ex: novo hospital entrou).

    Registros já classificados (HIGH/NORMAL/LOW) não são tocados — seus snapshots
    canonical_ref_low/high preservam as referências usadas na ingestão original.

    Deve ser executado após persist() — usa analyte_references já populada.
    """
    from sqlalchemy import text

    log.info("Backfill de classifications em exam_records ...")

    result = conn.execute(text("""
        UPDATE metrics.exam_records e
        SET
            canonical_ref_low  = r.ref_low,
            canonical_ref_high = r.ref_high,
            classification = CASE
                WHEN r.ref_low = 0 AND r.ref_high = 0 THEN 'NO_REF'
                WHEN e.value < r.ref_low               THEN 'LOW'
                WHEN e.value > r.ref_high              THEN 'HIGH'
                ELSE 'NORMAL'
            END
        FROM knowledge.analyte_references r
        WHERE r.canonical = e.analyte
          AND r.sex IS NULL
          AND (e.classification IS NULL OR e.classification = 'NO_REF')
    """))
    conn.commit()

    updated = result.rowcount
    log.info(f"  {updated:,} registros classificados.")
    return updated


def print_report(entries: list[dict]) -> None:
    print(f"\n{'Analito':<40} {'N hosp':>7}  {'ref_low':>9}  {'ref_high':>9}")
    print("─" * 72)
    for e in entries:
        print(
            f"{e['canonical']:<40} {e['n_hospitals']:>7}  "
            f"{e['ref_low']:>9.3f}  {e['ref_high']:>9.3f}"
        )
    print(f"\nTotal: {len(entries)} analitos\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",       action="store_true", help="Calcula mas não grava")
    parser.add_argument("--min-hospitals", type=int, default=1,
                        help="Mínimo de hospitais para incluir o analito (default: 1)")
    parser.add_argument("--no-backfill",   action="store_true",
                        help="Não preenche classification em exam_records")
    args = parser.parse_args()

    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        log.error("FL_DB_URL não definida.")
        sys.exit(1)

    from sqlalchemy import create_engine
    engine = create_engine(db_url)

    with engine.connect() as conn:
        log.info("Calculando referências canônicas ...")
        entries = compute(conn, min_hospitals=args.min_hospitals)
        log.info(f"  {len(entries)} analitos encontrados.")
        print_report(entries)

        if args.dry_run:
            log.info("Modo dry-run — nenhum dado gravado.")
            return

        n = persist(conn, entries)
        log.info(f"  {n} entradas gravadas em knowledge.analyte_references.")

        if not args.no_backfill:
            backfill_classifications(conn)


if __name__ == "__main__":
    main()
