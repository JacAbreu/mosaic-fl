"""clinical_mixin.py — Dados clínicos de série temporal: risco, exames e desfechos."""
from datetime import date as _date
from typing import Optional

from sqlalchemy import func, insert, select


class _ClinicalMixin:
    """Requer os atributos definidos em _PatientDBCore (_engine, _risk, _exams, _outcomes)."""

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
                "patient_id":         patient_id,
                "analyte":            e["analyte"],
                "date":               e["date"] if isinstance(e["date"], _date) else _date.fromisoformat(e["date"]),
                "value":              e["value"],
                "phase":              e["phase"],
                "ref_low":            e.get("ref_low",            0.0),
                "ref_high":           e.get("ref_high",           0.0),
                "origin":             e.get("origin"),
                "exam_group":         e.get("exam_group"),
                "value_text":         e.get("value_text"),
                "unit":               e.get("unit"),
                "attendance_id":      e.get("attendance_id"),
                "canonical_ref_low":  e.get("canonical_ref_low"),
                "canonical_ref_high": e.get("canonical_ref_high"),
                "classification":     e.get("classification"),
            }
            for e in exams
        ]
        with self._engine.begin() as conn:
            conn.execute(insert(self._exams), rows)

    def add_exams_bulk(self, rows: list[dict]) -> None:
        """Insere múltiplas linhas de exames em uma única transação.

        Cada elemento de `rows` deve conter 'patient_id' junto com os demais campos —
        use quando o batch já tem múltiplos pacientes misturados (carga em streaming).
        """
        if not rows:
            return
        db_rows = [
            {
                "patient_id":         r["patient_id"],
                "analyte":            r["analyte"],
                "date":               r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"]),
                "value":              r["value"],
                "phase":              r["phase"],
                "ref_low":            r.get("ref_low",            0.0),
                "ref_high":           r.get("ref_high",           0.0),
                "origin":             r.get("origin"),
                "exam_group":         r.get("exam_group"),
                "value_text":         r.get("value_text"),
                "unit":               r.get("unit"),
                "attendance_id":      r.get("attendance_id"),
                "canonical_ref_low":  r.get("canonical_ref_low"),
                "canonical_ref_high": r.get("canonical_ref_high"),
                "classification":     r.get("classification"),
            }
            for r in rows
        ]
        with self._engine.begin() as conn:
            conn.execute(insert(self._exams), db_rows)

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
                self._exams.c.canonical_ref_low,
                self._exams.c.canonical_ref_high,
                self._exams.c.classification,
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
