"""transactional_mixin.py — Variantes explícitas de transação (recebem `conn`), para operações atômicas multi-tabela."""
import contextlib
from datetime import date as _date
from typing import Optional

from sqlalchemy import insert, select


class _TransactionalMixin:
    """Requer os atributos definidos em _PatientDBCore e os _stmt_upsert_* de _PatientDBCore."""

    @contextlib.contextmanager
    def begin(self):
        """Abre uma transação explícita. Use com os métodos _tx para atomicidade."""
        with self._engine.begin() as conn:
            yield conn

    def upsert_patient_tx(self, conn, patient_id: str, sex: str, age: float,
                          birth_year: Optional[int] = None, state_code: Optional[str] = None,
                          hospital_id: Optional[str] = None, municipality: Optional[str] = None,
                          cep_prefix: Optional[str] = None) -> None:
        conn.execute(self._stmt_upsert_patient(
            patient_id, sex, age, birth_year, state_code, hospital_id, municipality, cep_prefix,
        ))

    def add_exams_tx(self, conn, patient_id: str, exams: list[dict]) -> None:
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
        conn.execute(insert(self._exams), rows)

    def get_exams_tx(self, conn, patient_id: str) -> list[dict]:
        stmt = (
            select(
                self._exams.c.analyte, self._exams.c.exam_group, self._exams.c.date,
                self._exams.c.value, self._exams.c.value_text, self._exams.c.phase,
                self._exams.c.origin, self._exams.c.unit, self._exams.c.ref_low,
                self._exams.c.ref_high, self._exams.c.attendance_id,
                self._exams.c.canonical_ref_low, self._exams.c.canonical_ref_high,
                self._exams.c.classification,
            )
            .where(self._exams.c.patient_id == patient_id)
            .order_by(self._exams.c.date, self._exams.c.id)
        )
        rows = conn.execute(stmt).mappings()
        return [
            dict(r) | {"date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"])}
            for r in rows
        ]

    def get_patient_tx(self, conn, patient_id: str) -> Optional[dict]:
        stmt = select(self._patients).where(self._patients.c.patient_id == patient_id)
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    def add_risk_tx(self, conn, patient_id: str, date_val, risk_score: float) -> None:
        d = date_val if isinstance(date_val, _date) else _date.fromisoformat(date_val)
        conn.execute(insert(self._risk).values(patient_id=patient_id, date=d, risk_score=risk_score))

    def get_risk_history_tx(self, conn, patient_id: str) -> list[dict]:
        stmt = (
            select(self._risk.c.date, self._risk.c.risk_score)
            .where(self._risk.c.patient_id == patient_id)
            .order_by(self._risk.c.date, self._risk.c.id)
        )
        return [
            {
                "date": r["date"] if isinstance(r["date"], _date) else _date.fromisoformat(r["date"]),
                "risk_score": r["risk_score"],
            }
            for r in conn.execute(stmt).mappings()
        ]

    def set_export_path_tx(self, conn, patient_id: str, path: str) -> None:
        conn.execute(self._stmt_upsert_export(patient_id, path))
