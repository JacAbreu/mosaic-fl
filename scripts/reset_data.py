"""
reset_data.py
Trunca as tabelas de dados do MOSAIC-FL para permitir uma recarga completa.

O que é apagado (dados clínicos carregados):
  metrics.exam_records
  metrics.clinical_outcomes
  metrics.risk_history
  clinical.attendances
  clinical.export_paths
  clinical.patients

O que NÃO é apagado (curadoria do operador):
  knowledge.term_dictionary    — aliases revisados e ativados
  knowledge.analyte_references — referências canônicas computadas

Sequência de TRUNCATE respeita as dependências de FK (ordem inversa de inserção).

Uso:
    python scripts/reset_data.py --db-url postgresql://...
    python scripts/reset_data.py --db-url postgresql://... --confirm
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_TABLES_IN_ORDER = [
    # Ordem reversa de FK: dependentes primeiro
    "metrics.exam_records",
    "metrics.clinical_outcomes",
    "metrics.risk_history",
    "clinical.export_paths",
    "clinical.attendances",
    "clinical.patients",
]

_PRESERVED = [
    "knowledge.term_dictionary",
    "knowledge.analyte_references",
]


def _count(conn, table: str) -> int:
    from sqlalchemy import text
    schema, tbl = table.split(".")
    try:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar_one()
    except Exception:
        return -1


def reset(db_url: str, confirm: bool) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    with engine.connect() as conn:
        log.info("\nContagem atual antes do reset:")
        for table in _TABLES_IN_ORDER:
            n = _count(conn, table)
            label = f"{n:>12,}" if n >= 0 else "     (erro)"
            log.info("  %-40s %s", table, label)

        log.info("\nPreservadas (não serão apagadas):")
        for table in _PRESERVED:
            n = _count(conn, table)
            label = f"{n:>12,}" if n >= 0 else "     (erro)"
            log.info("  %-40s %s", table, label)

        if not confirm:
            log.warning(
                "\nNenhum dado foi apagado. "
                "Execute com --confirm para confirmar o reset."
            )
            return

        log.info("\nTruncando tabelas...")
        for table in _TABLES_IN_ORDER:
            try:
                conn.execute(text(f"TRUNCATE {table} RESTART IDENTITY CASCADE"))
                log.info("  ✓ %s", table)
            except Exception as e:
                log.error("  ✗ %s — %s", table, e)
                conn.rollback()
                sys.exit(1)

        conn.commit()
        log.info("\nReset concluído. knowledge.* preservado.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("FL_DB_URL"),
        help="SQLAlchemy database URL (default: FL_DB_URL env var).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Necessário para executar o TRUNCATE. Sem este flag, apenas exibe contagens.",
    )
    args = parser.parse_args()

    if not args.db_url:
        log.error("FL_DB_URL não definida. Use --db-url ou exporte a variável.")
        sys.exit(1)

    reset(args.db_url, confirm=args.confirm)


if __name__ == "__main__":
    main()
