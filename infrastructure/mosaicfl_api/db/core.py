"""core.py — Inicialização de PatientDB e statement builders (dialect-aware upsert) compartilhados pelos mixins."""
import os
from datetime import date as _date
from pathlib import Path
from typing import Optional, Union

from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .engine import _make_engine
from .schema import _build_tables

_DEFAULT_URL = os.getenv("FL_DB_URL", "sqlite:///data/mosaicfl_api.db")


class _PatientDBCore:
    """
    Inicialização compartilhada + helpers de upsert dialect-aware.

    PostgreSQL: clinical schema for registry/config, metrics schema for time-series.
    SQLite:     all tables without schema prefix (dev/tests — identical interface).
    """

    def __init__(self, db_url: Union[str, Path] = _DEFAULT_URL):
        if isinstance(db_url, Path):
            db_url = f"sqlite:///{db_url}"
        url = str(db_url)
        self._is_pg = url.startswith("postgresql")

        if not self._is_pg:
            # sqlite:///path/to/file.db → extrai o path e cria o diretório se necessário
            sqlite_path = Path(url.replace("sqlite:///", "", 1))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = _make_engine(url)
        (
            self._meta,
            self._patients,
            self._attendances,
            self._exports,
            self._risk,
            self._exams,
            self._outcomes,
            self._predicted_outcomes,
            self._outcome_feedback,
        ) = _build_tables(self._is_pg)

        if not self._is_pg:
            self._meta.create_all(self._engine)

        import logging
        logging.getLogger(__name__).info(
            "db_initialized", extra={"backend": "postgresql" if self._is_pg else "sqlite"}
        )

    # ── upsert helpers (dialect-aware) ────────────────────────────────────

    def _stmt_upsert_patient(self, patient_id: str, sex: str, age: float,
                              birth_year: Optional[int], state_code: Optional[str],
                              hospital_id: Optional[str], municipality: Optional[str],
                              cep_prefix: Optional[str]):
        vals = {
            "patient_id":   patient_id,
            "sex":          sex,
            "age":          age,
            "birth_year":   birth_year,
            "state_code":   state_code,
            "hospital_id":  hospital_id,
            "municipality": municipality,
            "cep_prefix":   cep_prefix,
        }
        if self._is_pg:
            return pg_insert(self._patients).values(**vals).on_conflict_do_update(
                index_elements=["patient_id"],
                set_={"municipality": municipality, "cep_prefix": cep_prefix},
            )
        return insert(self._patients).prefix_with("OR IGNORE").values(**vals)

    def _stmt_upsert_attendance(self, attendance_id: str, patient_id: str,
                                 hospital_id: Optional[str], attended_at: _date,
                                 attendance_type: Optional[str], specialty: Optional[str],
                                 clinic_id: Optional[str],
                                 suspected_diagnosis: Optional[str],
                                 confirmed_diagnosis: Optional[str]):
        vals = {
            "attendance_id":       attendance_id,
            "patient_id":          patient_id,
            "hospital_id":         hospital_id,
            "attended_at":         attended_at,
            "attendance_type":     attendance_type,
            "specialty":           specialty,
            "clinic_id":           clinic_id,
            "suspected_diagnosis": suspected_diagnosis,
            "confirmed_diagnosis": confirmed_diagnosis,
        }
        if self._is_pg:
            return pg_insert(self._attendances).values(**vals).on_conflict_do_update(
                index_elements=["attendance_id"],
                set_={
                    "clinic_id":           clinic_id,
                    "suspected_diagnosis": suspected_diagnosis,
                    "confirmed_diagnosis": confirmed_diagnosis,
                },
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
