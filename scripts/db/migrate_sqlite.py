#!/usr/bin/env python3
"""
migrate_sqlite.py — Migra PatientDB do SQLite para PostgreSQL.

Uso:
    python scripts/db/migrate_sqlite.py \\
        --sqlite data/mosaicfl_api.db \\
        --pg "postgresql://mosaicfl:senha@localhost:5432/mosaicfl"

Ou via variáveis de ambiente:
    FL_SQLITE_PATH=data/mosaicfl_api.db \\
    FL_DB_URL=postgresql://... \\
    python scripts/db/migrate_sqlite.py

Idempotente: pode ser executado múltiplas vezes sem duplicar dados.
"""
import argparse
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_api.db import PatientDB

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def migrate(sqlite_path: str, pg_url: str) -> None:
    if not Path(sqlite_path).exists():
        log.error(f"Arquivo SQLite não encontrado: {sqlite_path}")
        sys.exit(1)

    log.info(f"Origem:  {sqlite_path}")
    log.info(f"Destino: {pg_url.split('@')[-1]}")  # omite credenciais do log

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    dst = PatientDB(pg_url)

    # 1. Pacientes (deve vir primeiro — FK de risk_history referencia patients)
    patients = src.execute("SELECT * FROM patients").fetchall()
    log.info(f"  pacientes:      {len(patients)}")
    for p in patients:
        dst.upsert_patient(p["patient_id"], p["sex"], p["age"])

    # 2. Histórico de risco (ordem por id preserva cronologia)
    risks = src.execute("SELECT * FROM risk_history ORDER BY id").fetchall()
    log.info(f"  risk_history:   {len(risks)}")
    for r in risks:
        dst.add_risk(r["patient_id"], r["date"], r["risk_score"])

    # 3. Exames (em batch por paciente)
    exams = src.execute("SELECT * FROM exam_records ORDER BY id").fetchall()
    log.info(f"  exam_records:   {len(exams)}")
    by_patient: dict = defaultdict(list)
    for e in exams:
        by_patient[e["patient_id"]].append(dict(e))
    for patient_id, batch in by_patient.items():
        dst.add_exams(patient_id, batch)

    # 4. Caminhos de exportação
    exports = src.execute("SELECT * FROM export_paths").fetchall()
    log.info(f"  export_paths:   {len(exports)}")
    for e in exports:
        dst.set_export_path(e["patient_id"], e["export_path"])

    src.close()
    log.info("Migração concluída com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migra PatientDB SQLite → PostgreSQL")
    parser.add_argument(
        "--sqlite",
        default=os.getenv("FL_SQLITE_PATH", "data/mosaicfl_api.db"),
        help="Caminho do arquivo SQLite de origem",
    )
    parser.add_argument(
        "--pg",
        default=os.getenv("FL_DB_URL", ""),
        help="URL PostgreSQL de destino",
    )
    args = parser.parse_args()

    if not args.pg or not args.pg.startswith("postgresql"):
        log.error("--pg deve ser uma URL PostgreSQL válida (postgresql://...)")
        sys.exit(1)

    migrate(args.sqlite, args.pg)
