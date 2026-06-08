"""
db.py
Persistence layer for mosaicfl_api.

Schemas:
  clinical  — patient registry, attendances, export paths, FL config (PostgreSQL pure)
  metrics   — time-series exam records, clinical outcomes, risk history (TimescaleDB)

Backend selected by FL_DB_URL:
  postgresql://mosaicfl:senha@localhost:5432/mosaicfl  → PostgreSQL
  sqlite:///data/mosaicfl_api.db                       → SQLite (dev/tests)

Constructor accepts Path for backwards compatibility:
  PatientDB(Path("foo.db"))  →  SQLite at foo.db
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
# Table definitions
# ---------------------------------------------------------------------------

def _build_tables(is_pg: bool):
    clinical = "clinical" if is_pg else None
    metrics  = "metrics"  if is_pg else None

    meta = sa.MetaData()

    patients = sa.Table(
        "patients", meta,
        sa.Column("patient_id",  sa.Text,     primary_key=True),
        sa.Column("sex",         sa.Text,     nullable=False, server_default=sa.text("'M'")),
        sa.Column("age",         sa.Float,    nullable=False, server_default=sa.text("0.0")),
        sa.Column("birth_year",  sa.SmallInteger),
        sa.Column("state_code",  sa.String(2)),
        sa.Column("hospital_id", sa.Text),
        schema=clinical,
    )
    attendances = sa.Table(
        "attendances", meta,
        sa.Column("attendance_id",   sa.Text, primary_key=True),
        sa.Column("patient_id",      sa.Text, nullable=False),
        sa.Column("hospital_id",     sa.Text),
        sa.Column("attended_at",     sa.Date, nullable=False),
        sa.Column("attendance_type", sa.Text),
        sa.Column("specialty",       sa.Text),
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
        sa.Column("patient_id", sa.Text,    nullable=False),
        sa.Column("date",       sa.Date,    nullable=False),
        sa.Column("risk_score", sa.Float,   nullable=False),
        schema=metrics,
    )
    exam_records = sa.Table(
        "exam_records", meta,
        sa.Column("id",            sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("patient_id",    sa.Text,  nullable=False),
        sa.Column("analyte",       sa.Text,  nullable=False),
        sa.Column("date",          sa.Date,  nullable=False),
        sa.Column("value",         sa.Float, nullable=False),
        sa.Column("phase",         sa.Text,  nullable=False),
        sa.Column("ref_low",       sa.Float, server_default=sa.text("0.0")),
        sa.Column("ref_high",      sa.Float, server_default=sa.text("0.0")),
        sa.Column("origin",        sa.Text),
        sa.Column("exam_group",    sa.Text),
        sa.Column("value_text",    sa.Text),
        sa.Column("unit",          sa.Text),
        sa.Column("attendance_id", sa.Text),
        schema=metrics,
    )
    clinical_outcomes = sa.Table(
        "clinical_outcomes", meta,
        sa.Column("id",            sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("patient_id",    sa.Text,         nullable=False),
        sa.Column("attendance_id", sa.Text),
        sa.Column("outcome_at",    sa.Date,         nullable=False),
        sa.Column("outcome_text",  sa.Text,         nullable=False),
        sa.Column("outcome_class", sa.SmallInteger, nullable=False),
        schema=metrics,
    )

    return meta, patients, attendances, export_paths, risk_history, exam_records, clinical_outcomes


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
    Patient data access layer.

    PostgreSQL: clinical schema for registry/config, metrics schema for time-series.
    SQLite:     all tables without schema prefix (dev/tests — identical interface).
    """

    def __init__(self, db_url: Union[str, Path] = _DEFAULT_URL):
        if isinstance(db_url, Path):
            db_url = f"sqlite:///{db_url}"
        url = str(db_url)
        self._is_pg = url.startswith("postgresql")
        self._engine = _make_engine(url)
        (
            self._meta,
            self._patients,
            self._attendances,
            self._exports,
            self._risk,
            self._exams,
            self._outcomes,
        ) = _build_tables(self._is_pg)

        if not self._is_pg:
            self._meta.create_all(self._engine)

        logger.info("db_initialized", extra={"backend": "postgresql" if self._is_pg else "sqlite"})

    # ── upsert helpers (dialect-aware) ────────────────────────────────────

    def _stmt_upsert_patient(self, patient_id: str, sex: str, age: float,
                              birth_year: Optional[int], state_code: Optional[str],
                              hospital_id: Optional[str]):
        vals = {
            "patient_id":  patient_id,
            "sex":         sex,
            "age":         age,
            "birth_year":  birth_year,
            "state_code":  state_code,
            "hospital_id": hospital_id,
        }
        if self._is_pg:
            return pg_insert(self._patients).values(**vals).on_conflict_do_nothing(
                index_elements=["patient_id"]
            )
        return insert(self._patients).prefix_with("OR IGNORE").values(**vals)

    def _stmt_upsert_attendance(self, attendance_id: str, patient_id: str,
                                 hospital_id: Optional[str], attended_at: _date,
                                 attendance_type: Optional[str], specialty: Optional[str]):
        vals = {
            "attendance_id":   attendance_id,
            "patient_id":      patient_id,
            "hospital_id":     hospital_id,
            "attended_at":     attended_at,
            "attendance_type": attendance_type,
            "specialty":       specialty,
        }
        if self._is_pg:
            return pg_insert(self._attendances).values(**vals).on_conflict_do_nothing(
                index_elements=["attendance_id"]
            )
        return insert(self._attendances).prefix_with("OR IGNORE").values(**vals)

    def _stmt_upsert_export(self, patient_id: str, path: str):
        vals = {"patient_id": patient_id, "export_path": path}
        if self._is_pg:
            return pg_insert(self._exports).values(**vals).on_conflict_do_update(
                index_elements=["patient_id"],
                set_={"export_path": path},
            )
        return insert(self._exports).prefix_with("OR REPLACE").values(**vals)

    # ── patients ──────────────────────────────────────────────────────────

    def upsert_patient(
        self,
        patient_id: str,
        sex: str,
        age: float,
        birth_year: Optional[int] = None,
        state_code: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_patient(
                patient_id, sex, age, birth_year, state_code, hospital_id
            ))

    def patient_exists(self, patient_id: str) -> bool:
        stmt = select(self._patients.c.patient_id).where(
            self._patients.c.patient_id == patient_id
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).first() is not None

    def list_patients(self) -> list[dict]:
        p_ref  = "clinical.patients"    if self._is_pg else "patients"
        rh_ref = "metrics.risk_history" if self._is_pg else "risk_history"
        with self._engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT p.patient_id, p.sex, p.age, p.birth_year, p.state_code, p.hospital_id,
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

    # ── attendances ───────────────────────────────────────────────────────

    def upsert_attendance(
        self,
        attendance_id: str,
        patient_id: str,
        attended_at: "str | _date",
        hospital_id: Optional[str] = None,
        attendance_type: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> None:
        d = attended_at if isinstance(attended_at, _date) else _date.fromisoformat(attended_at)
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_attendance(
                attendance_id, patient_id, hospital_id, d, attendance_type, specialty
            ))

    # ── risk history ──────────────────────────────────────────────────────

    def add_risk(self, patient_id: str, date_val: "str | _date", risk_score: float) -> None:
        d = date_val if isinstance(date_val, _date) else _date.fromisoformat(date_val)
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._risk).values(patient_id=patient_id, date=d, risk_score=risk_score)
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
                {
                    "date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"]),
                    "risk_score": r["risk_score"],
                }
                for r in rows
            ]

    # ── exam records ──────────────────────────────────────────────────────

    def add_exams(self, patient_id: str, exams: list[dict]) -> None:
        rows = [
            {
                "patient_id":    patient_id,
                "analyte":       e["analyte"],
                "date":          e["date"] if isinstance(e["date"], _date) else _date.fromisoformat(e["date"]),
                "value":         e["value"],
                "phase":         e["phase"],
                "ref_low":       e.get("ref_low",       0.0),
                "ref_high":      e.get("ref_high",      0.0),
                "origin":        e.get("origin"),
                "exam_group":    e.get("exam_group"),
                "value_text":    e.get("value_text"),
                "unit":          e.get("unit"),
                "attendance_id": e.get("attendance_id"),
            }
            for e in exams
        ]
        with self._engine.begin() as conn:
            conn.execute(insert(self._exams), rows)

    def get_exams(self, patient_id: str) -> list[dict]:
        stmt = (
            select(
                self._exams.c.analyte,
                self._exams.c.exam_group,
                self._exams.c.date,
                self._exams.c.value,
                self._exams.c.value_text,
                self._exams.c.phase,
                self._exams.c.origin,
                self._exams.c.unit,
                self._exams.c.ref_low,
                self._exams.c.ref_high,
                self._exams.c.attendance_id,
            )
            .where(self._exams.c.patient_id == patient_id)
            .order_by(self._exams.c.date, self._exams.c.id)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings()
            return [
                dict(r) | {
                    "date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"])
                }
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

    # ── clinical outcomes ─────────────────────────────────────────────────

    def add_clinical_outcome(
        self,
        patient_id: str,
        outcome_at: "str | _date",
        outcome_text: str,
        outcome_class: int,
        attendance_id: Optional[str] = None,
    ) -> None:
        d = outcome_at if isinstance(outcome_at, _date) else _date.fromisoformat(outcome_at)
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._outcomes).values(
                    patient_id=patient_id,
                    attendance_id=attendance_id,
                    outcome_at=d,
                    outcome_text=outcome_text,
                    outcome_class=outcome_class,
                )
            )

    def get_clinical_outcomes(self, patient_id: str) -> list[dict]:
        stmt = (
            select(
                self._outcomes.c.attendance_id,
                self._outcomes.c.outcome_at,
                self._outcomes.c.outcome_text,
                self._outcomes.c.outcome_class,
            )
            .where(self._outcomes.c.patient_id == patient_id)
            .order_by(self._outcomes.c.outcome_at, self._outcomes.c.id)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings()
            return [
                dict(r) | {
                    "outcome_at": r["outcome_at"] if isinstance(r["outcome_at"], _date)
                                  else _date.fromisoformat(r["outcome_at"])
                }
                for r in rows
            ]

    # ── export paths ──────────────────────────────────────────────────────

    def set_export_path(self, patient_id: str, path: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_export(patient_id, path))

    def get_export_path(self, patient_id: str) -> Optional[str]:
        stmt = select(self._exports.c.export_path).where(
            self._exports.c.patient_id == patient_id
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one_or_none()
