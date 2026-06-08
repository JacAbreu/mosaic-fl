"""
db.py
Persistência para o mosaicfl_api.

Schemas:
  clinical  — cadastro de pacientes e export_paths (PostgreSQL puro)
  metrics   — risk_history e exam_records como hypertables TimescaleDB

Backend selecionado por FL_DB_URL:
  postgresql://mosaicfl:senha@localhost:5432/mosaicfl  → PostgreSQL
  sqlite:///data/mosaicfl_api.db                       → SQLite (dev/testes)

O construtor aceita Path para compatibilidade retroativa:
  PatientDB(Path("foo.db"))  →  SQLite em foo.db
"""
import logging
import os
from datetime import date as _date
from pathlib import Path
from typing import Optional, Union

import sqlalchemy as sa
from sqlalchemy import func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)

_DEFAULT_URL = os.getenv("FL_DB_URL", "sqlite:///data/mosaicfl_api.db")


# ---------------------------------------------------------------------------
# Definição de tabelas
# ---------------------------------------------------------------------------

def _build_tables(is_pg: bool):
    """
    Retorna MetaData + tabelas com schemas corretos por backend.

    PostgreSQL: clinical.patients, metrics.risk_history, metrics.exam_records, clinical.export_paths
    SQLite:     sem schema (SQLite não suporta schemas PostgreSQL)
    """
    clinical = "clinical" if is_pg else None
    metrics  = "metrics"  if is_pg else None

    meta = sa.MetaData()

    patients = sa.Table(
        "patients", meta,
        sa.Column("patient_id", sa.Text,  primary_key=True),
        sa.Column("sex",        sa.Text,  nullable=False, server_default=sa.text("'M'")),
        sa.Column("age",        sa.Float, nullable=False, server_default=sa.text("0.0")),
        schema=clinical,
    )
    export_paths = sa.Table(
        "export_paths", meta,
        sa.Column("patient_id",  sa.Text, primary_key=True),
        sa.Column("export_path", sa.Text, nullable=False),
        schema=clinical,
    )
    risk_history = sa.Table(
        "risk_history", meta,
        sa.Column("id",         sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("patient_id", sa.Text,  nullable=False),
        sa.Column("date",       sa.Date,  nullable=False),
        sa.Column("risk_score", sa.Float, nullable=False),
        schema=metrics,
    )
    exam_records = sa.Table(
        "exam_records", meta,
        sa.Column("id",           sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("patient_id",   sa.Text,  nullable=False),
        sa.Column("exam_name",    sa.Text,  nullable=False),
        sa.Column("date",         sa.Date,  nullable=False),
        sa.Column("value",        sa.Float, nullable=False),
        sa.Column("phase",        sa.Text,  nullable=False),
        sa.Column("ref_low",      sa.Float, server_default=sa.text("0.0")),
        sa.Column("ref_high",     sa.Float, server_default=sa.text("0.0")),
        sa.Column("sex_ref_low",  sa.Float, server_default=sa.text("0.0")),
        sa.Column("sex_ref_high", sa.Float, server_default=sa.text("0.0")),
        schema=metrics,
    )

    return meta, patients, export_paths, risk_history, exam_records


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(url: str) -> sa.Engine:
    if url.startswith("postgresql"):
        return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    from sqlalchemy.pool import StaticPool
    return sa.create_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


# ---------------------------------------------------------------------------
# PatientDB
# ---------------------------------------------------------------------------

class PatientDB:
    """
    Acesso a dados de pacientes.

    PostgreSQL: cadastro em schema clinical, séries temporais em schema metrics.
    SQLite:     todas as tabelas sem schema (dev/testes — interface idêntica).
    """

    def __init__(self, db_url: Union[str, Path] = _DEFAULT_URL):
        if isinstance(db_url, Path):
            db_url = f"sqlite:///{db_url}"
        url = str(db_url)
        self._is_pg = url.startswith("postgresql")
        self._engine = _make_engine(url)
        self._meta, self._patients, self._exports, self._risk, self._exams = _build_tables(self._is_pg)

        if not self._is_pg:
            # SQLite: cria tabelas automaticamente (PostgreSQL usa init.sql)
            self._meta.create_all(self._engine)

        logger.info("db_initialized", extra={"backend": "postgresql" if self._is_pg else "sqlite"})

    # ── helpers de upsert (dialect-aware) ─────────────────────────────────

    def _stmt_upsert_patient(self, patient_id: str, sex: str, age: float):
        vals = {"patient_id": patient_id, "sex": sex, "age": age}
        if self._is_pg:
            return pg_insert(self._patients).values(**vals).on_conflict_do_nothing(
                index_elements=["patient_id"]
            )
        return insert(self._patients).prefix_with("OR IGNORE").values(**vals)

    def _stmt_upsert_export(self, patient_id: str, path: str):
        vals = {"patient_id": patient_id, "export_path": path}
        if self._is_pg:
            return pg_insert(self._exports).values(**vals).on_conflict_do_update(
                index_elements=["patient_id"],
                set_={"export_path": path},
            )
        return insert(self._exports).prefix_with("OR REPLACE").values(**vals)

    # ── pacientes ──────────────────────────────────────────────────────────

    def upsert_patient(self, patient_id: str, sex: str, age: float) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_patient(patient_id, sex, age))

    def patient_exists(self, patient_id: str) -> bool:
        stmt = select(self._patients.c.patient_id).where(
            self._patients.c.patient_id == patient_id
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).first() is not None

    def list_patients(self) -> list[dict]:
        p_ref  = f"clinical.patients"     if self._is_pg else "patients"
        rh_ref = f"metrics.risk_history"  if self._is_pg else "risk_history"
        with self._engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT p.patient_id, p.sex, p.age,
                       r.risk_score AS latest_risk,
                       r.date       AS latest_date
                FROM {p_ref} p
                LEFT JOIN {rh_ref} r
                    ON r.id = (
                        SELECT id FROM {rh_ref}
                        WHERE patient_id = p.patient_id
                        ORDER BY date DESC, id DESC
                        LIMIT 1
                    )
                ORDER BY p.patient_id
            """)).mappings().all()
        return [dict(row) for row in rows]

    # ── histórico de risco ─────────────────────────────────────────────────

    def add_risk(self, patient_id: str, date_val: "str | _date", risk_score: float) -> None:
        d = date_val if isinstance(date_val, _date) else _date.fromisoformat(date_val)
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._risk).values(
                    patient_id=patient_id, date=d, risk_score=risk_score
                )
            )

    def get_risk_history(self, patient_id: str) -> list[dict]:
        stmt = (
            select(self._risk.c.date, self._risk.c.risk_score)
            .where(self._risk.c.patient_id == patient_id)
            .order_by(self._risk.c.date, self._risk.c.id)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings()
            return [
                {"date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"]),
                 "risk_score": r["risk_score"]}
                for r in rows
            ]

    # ── exames ─────────────────────────────────────────────────────────────

    def add_exams(self, patient_id: str, exams: list[dict]) -> None:
        rows = [
            {
                "patient_id":   patient_id,
                "exam_name":    e["exam_name"],
                "date":         e["date"] if isinstance(e["date"], _date) else _date.fromisoformat(e["date"]),
                "value":        e["value"],
                "phase":        e["phase"],
                "ref_low":      e.get("ref_low",      0.0),
                "ref_high":     e.get("ref_high",     0.0),
                "sex_ref_low":  e.get("sex_ref_low",  0.0),
                "sex_ref_high": e.get("sex_ref_high", 0.0),
            }
            for e in exams
        ]
        with self._engine.begin() as conn:
            conn.execute(insert(self._exams), rows)

    def get_exams(self, patient_id: str) -> list[dict]:
        stmt = (
            select(
                self._exams.c.exam_name,
                self._exams.c.date,
                self._exams.c.value,
                self._exams.c.phase,
                self._exams.c.ref_low,
                self._exams.c.ref_high,
                self._exams.c.sex_ref_low,
                self._exams.c.sex_ref_high,
            )
            .where(self._exams.c.patient_id == patient_id)
            .order_by(self._exams.c.date, self._exams.c.id)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings()
            return [
                dict(r) | {"date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"])}
                for r in rows
            ]

    def exam_count(self, patient_id: str) -> int:
        stmt = (
            select(func.count())
            .select_from(self._exams)
            .where(self._exams.c.patient_id == patient_id)
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()

    # ── caminhos de exportação ─────────────────────────────────────────────

    def set_export_path(self, patient_id: str, path: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_export(patient_id, path))

    def get_export_path(self, patient_id: str) -> Optional[str]:
        stmt = select(self._exports.c.export_path).where(
            self._exports.c.patient_id == patient_id
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one_or_none()
