"""
state_store.py — persistência do estado do scheduler em SQLite.

SQLite é usado como solução pragmática e de baixo custo operacional.
A interface pública (load / save / record_round) é agnóstica ao banco,
o que facilita a migração futura para PostgreSQL ou outro SGBD:
basta trocar _connect() e _SCHEMA pelo driver e dialect correspondentes.

Configuração:
    FL_SCHEDULER_DB=/app/data/scheduler.db  (default)
"""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schedule_state import SchedulerState

DEFAULT_DB_PATH = Path(os.getenv("FL_SCHEDULER_DB", "/app/data/scheduler.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduler_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS round_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num     INTEGER NOT NULL,
    accuracy      REAL,
    dispatched_at TEXT NOT NULL,
    success       INTEGER NOT NULL DEFAULT 1
);
"""


class SchedulerStateStore:
    """
    Persiste SchedulerState em SQLite.

    Substitui o SchedulerState.save/load baseado em JSON no diretório corrente.
    Para migrar para PostgreSQL: substitua _connect() por psycopg2/SQLAlchemy
    e ajuste _SCHEMA para a sintaxe do dialeto alvo.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── infraestrutura ────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        # WAL evita corrupção em file systems de rede (NFS, EFS) — obrigatório em K8s
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── API pública ───────────────────────────────────────────────────────────

    def load(self) -> SchedulerState:
        """Carrega o estado persistido. Retorna estado inicial se o banco estiver vazio."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM scheduler_state").fetchall()

        if not rows:
            return SchedulerState()

        data = {r["key"]: json.loads(r["value"]) for r in rows}
        valid = SchedulerState.__dataclass_fields__.keys()
        return SchedulerState(**{k: v for k, v in data.items() if k in valid})

    def save(self, state: SchedulerState) -> None:
        """Persiste todos os campos do SchedulerState."""
        now = datetime.now().isoformat()
        fields = {
            "last_run": state.last_run,
            "current_round": state.current_round,
            "total_rounds_completed": state.total_rounds_completed,
            "client_history": state.client_history,
            "accuracy_history": state.accuracy_history,
            "converged": state.converged,
            "convergence_round": state.convergence_round,
        }
        with self._connect() as conn:
            for key, value in fields.items():
                conn.execute(
                    "INSERT OR REPLACE INTO scheduler_state (key, value, updated_at)"
                    " VALUES (?, ?, ?)",
                    (key, json.dumps(value), now),
                )

    def record_round(
        self,
        round_num: int,
        accuracy: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Registra um round na tabela de histórico (auditoria)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO round_history (round_num, accuracy, dispatched_at, success)"
                " VALUES (?, ?, ?, ?)",
                (round_num, accuracy, datetime.now().isoformat(), int(success)),
            )
