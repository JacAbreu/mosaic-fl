"""
db.py
Persistência SQLite para o mosaicfl_api.

Pacientes, histórico de risco e exames sobrevivem a reinicios do serviço.
Usa sqlite3 (stdlib) com WAL mode para suportar leituras concorrentes.

Localização padrão: FL_DB_PATH (env) ou data/mosaicfl_api.db
"""
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

_DB_PATH = Path(os.getenv("FL_DB_PATH", "data/mosaicfl_api.db"))

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS patients (
    patient_id  TEXT PRIMARY KEY,
    sex         TEXT NOT NULL DEFAULT 'M',
    age         REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS risk_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    risk_score  REAL    NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS exam_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  TEXT    NOT NULL,
    exam_name   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    value       REAL    NOT NULL,
    phase       TEXT    NOT NULL,
    ref_low     REAL    DEFAULT 0.0,
    ref_high    REAL    DEFAULT 0.0,
    sex_ref_low REAL    DEFAULT 0.0,
    sex_ref_high REAL   DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS export_paths (
    patient_id   TEXT PRIMARY KEY,
    export_path  TEXT NOT NULL
);
"""


class PatientDB:
    """Thread-safe wrapper sobre SQLite para o estado de pacientes."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── pacientes ──────────────────────────────────────────────────────────

    def upsert_patient(self, patient_id: str, sex: str, age: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id, sex, age) VALUES (?, ?, ?)",
                (patient_id, sex, age),
            )

    def list_patients(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.patient_id, p.sex, p.age,
                       r.risk_score AS latest_risk,
                       r.date       AS latest_date
                FROM patients p
                LEFT JOIN risk_history r
                    ON r.id = (
                        SELECT id FROM risk_history
                        WHERE patient_id = p.patient_id
                        ORDER BY date DESC, id DESC
                        LIMIT 1
                    )
                ORDER BY p.patient_id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ── histórico de risco ─────────────────────────────────────────────────

    def add_risk(self, patient_id: str, date_str: str, risk_score: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO risk_history (patient_id, date, risk_score) VALUES (?, ?, ?)",
                (patient_id, date_str, risk_score),
            )

    def get_risk_history(self, patient_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, risk_score FROM risk_history WHERE patient_id = ? ORDER BY date, id",
                (patient_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── exames ─────────────────────────────────────────────────────────────

    def add_exams(self, patient_id: str, exams: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO exam_records
                    (patient_id, exam_name, date, value, phase,
                     ref_low, ref_high, sex_ref_low, sex_ref_high)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        patient_id,
                        e["exam_name"],
                        e["date"],
                        e["value"],
                        e["phase"],
                        e.get("ref_low", 0.0),
                        e.get("ref_high", 0.0),
                        e.get("sex_ref_low", 0.0),
                        e.get("sex_ref_high", 0.0),
                    )
                    for e in exams
                ],
            )

    def get_exams(self, patient_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT exam_name, date, value, phase,
                          ref_low, ref_high, sex_ref_low, sex_ref_high
                   FROM exam_records WHERE patient_id = ? ORDER BY date, id""",
                (patient_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def exam_count(self, patient_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM exam_records WHERE patient_id = ?", (patient_id,)
            ).fetchone()[0]

    # ── caminhos de exportação ─────────────────────────────────────────────

    def set_export_path(self, patient_id: str, path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO export_paths (patient_id, export_path) VALUES (?, ?)",
                (patient_id, path),
            )

    def get_export_path(self, patient_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT export_path FROM export_paths WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return row["export_path"] if row else None

    def patient_exists(self, patient_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone() is not None
