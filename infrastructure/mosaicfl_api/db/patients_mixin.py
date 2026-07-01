"""patients_mixin.py — CRUD de pacientes, atendimentos e paths de exportação (dados de registro, não série temporal)."""
from datetime import date as _date
from typing import Optional

from sqlalchemy import func, select, text


class _PatientsMixin:
    """Requer os atributos definidos em _PatientDBCore (_engine, _patients, _attendances, _exports, _is_pg)."""

    # ── patients ──────────────────────────────────────────────────────────

    def upsert_patient(
        self,
        patient_id: str,
        sex: str,
        age: float,
        birth_year: Optional[int] = None,
        state_code: Optional[str] = None,
        hospital_id: Optional[str] = None,
        municipality: Optional[str] = None,
        cep_prefix: Optional[str] = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_patient(
                patient_id, sex, age, birth_year, state_code, hospital_id,
                municipality, cep_prefix,
            ))

    def patient_exists(self, patient_id: str) -> bool:
        stmt = select(self._patients.c.patient_id).where(
            self._patients.c.patient_id == patient_id
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).first() is not None

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Retorna dados de um paciente ou None se não existir."""
        stmt = select(self._patients).where(self._patients.c.patient_id == patient_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    def count_patients(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(self._patients)).scalar_one()

    def list_patients(self, limit: int = 100, offset: int = 0) -> list[dict]:
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
                LIMIT :lim OFFSET :off
            """), {"lim": limit, "off": offset}).mappings().all()
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
        clinic_id: Optional[str] = None,
        suspected_diagnosis: Optional[str] = None,
        confirmed_diagnosis: Optional[str] = None,
    ) -> None:
        d = attended_at if isinstance(attended_at, _date) else _date.fromisoformat(attended_at)
        with self._engine.begin() as conn:
            conn.execute(self._stmt_upsert_attendance(
                attendance_id, patient_id, hospital_id, d, attendance_type, specialty,
                clinic_id, suspected_diagnosis, confirmed_diagnosis,
            ))

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
