"""
CheckpointStore — persistência de pesos do modelo federado.

Dois backends com interface idêntica:
  SQLiteCheckpointStore   — experimentos locais (sem servidor de banco)
  PostgreSQLCheckpointStore — produção/homologação (PostgreSQL puro, sem pgvector)

Migração futura para S3: substitui apenas os bytes (store → S3 key no banco).

Seleção automática via get_checkpoint_store():
  FL_DB_URL configurado → PostgreSQLCheckpointStore
  FL_DB_URL vazio       → SQLiteCheckpointStore

Rastreabilidade por treinamento (migration 011):
  register_training()  — cria 1 linha em fl_trainings antes do loop FL
  save()               — UPSERT: 1 checkpoint por treinamento (melhor rodada)
  load_best()          — filtra por training_id; sem cross-contamination entre runs
  complete_training()  — atualiza fl_trainings com resultado final
"""
import hashlib
import io
import logging
import sqlite3
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS fl_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round       INTEGER NOT NULL,
    accuracy    REAL    NOT NULL DEFAULT 0.0,
    loss        REAL    NOT NULL DEFAULT 0.0,
    model_bytes BLOB    NOT NULL,
    sha256      TEXT    NOT NULL,
    vocab_size  INTEGER NOT NULL DEFAULT 0,
    training_id INTEGER,
    created_at  TEXT    NOT NULL
)
"""

_CREATE_SQLITE_TRAININGS = """
CREATE TABLE IF NOT EXISTS fl_trainings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm     TEXT NOT NULL DEFAULT 'FedAvg',
    log_file      TEXT NOT NULL DEFAULT '',
    n_rounds_max  INTEGER NOT NULL DEFAULT 120,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    n_rounds_done INTEGER,
    best_round    INTEGER,
    best_accuracy REAL,
    converged     INTEGER
)
"""


def _model_version(state_dict: OrderedDict) -> str:
    """Fingerprint SHA-256 dos pesos — 12 hex chars. Mesmo modelo → mesmo hash."""
    h = hashlib.sha256()
    for v in state_dict.values():
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:12]


def _serialize(
    state_dict: OrderedDict,
    vocab: Dict[str, int],
    temperature: float = 1.0,
    checkpoint_round: int = 0,
) -> bytes:
    buf = io.BytesIO()
    torch.save(
        {
            "model_state":      state_dict,
            "vocab":            vocab,
            "temperature":      temperature,
            "checkpoint_round": checkpoint_round,
            "checkpoint_at":    datetime.now(timezone.utc).isoformat(),
            "model_version":    _model_version(state_dict),
        },
        buf,
    )
    return buf.getvalue()


def _deserialize(data: bytes) -> Dict:
    buf = io.BytesIO(data)
    # weights_only=False: checkpoint contém vocab (dict str→int), temperatura e metadados além de tensors
    return torch.load(buf, map_location="cpu", weights_only=False)


class CheckpointStore(ABC):
    """Interface para persistência de checkpoints federados."""

    @abstractmethod
    def register_training(
        self,
        algorithm: str = "FedAvg",
        log_file: str = "",
        n_rounds_max: int = 120,
    ) -> int:
        """Registra um novo treinamento antes do loop FL. Retorna training_id."""

    @abstractmethod
    def complete_training(
        self,
        training_id: int,
        n_rounds_done: int,
        best_round: int,
        best_accuracy: float,
        converged: bool,
    ) -> None:
        """Atualiza fl_trainings com resultado final ao término do loop FL."""

    @abstractmethod
    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
        temperature: float = 1.0,
        training_id: Optional[int] = None,
    ) -> None:
        """UPSERT do checkpoint: 1 linha por training_id (substitui quando Acc melhora)."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna {'model_state': OrderedDict, 'vocab': dict} do checkpoint mais recente, ou None."""

    @abstractmethod
    def load_best(self, training_id: Optional[int] = None) -> Optional[Dict]:
        """Retorna o checkpoint com maior acurácia do treinamento indicado.
        Se training_id=None, usa o comportamento legado (melhor global — evitar)."""


class SQLiteCheckpointStore(CheckpointStore):
    """Checkpoint store em SQLite — para experimentos locais."""

    def __init__(self, db_path: str = "checkpoints/experiment.db") -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQLITE_TRAININGS)
            conn.execute(_CREATE_SQLITE)
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS fl_checkpoints_training_id_uniq "
                    "ON fl_checkpoints (training_id) WHERE training_id IS NOT NULL"
                )
            except sqlite3.OperationalError:
                pass
        logger.info("sqlite_checkpoint_store_ready path=%s", db_path)

    def register_training(
        self,
        algorithm: str = "FedAvg",
        log_file: str = "",
        n_rounds_max: int = 120,
    ) -> int:
        started_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO fl_trainings (algorithm, log_file, n_rounds_max, started_at) "
                "VALUES (?, ?, ?, ?)",
                (algorithm, log_file, n_rounds_max, started_at),
            )
            training_id = cur.lastrowid
        logger.info("training_registered_sqlite id=%d algorithm=%s", training_id, algorithm)
        return training_id

    def complete_training(
        self,
        training_id: int,
        n_rounds_done: int,
        best_round: int,
        best_accuracy: float,
        converged: bool,
    ) -> None:
        completed_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE fl_trainings SET status='completed', completed_at=?, "
                "n_rounds_done=?, best_round=?, best_accuracy=?, converged=? "
                "WHERE id=?",
                (completed_at, n_rounds_done, best_round, best_accuracy, int(converged), training_id),
            )
        logger.info(
            "training_completed_sqlite id=%d best_round=%d best_accuracy=%.4f converged=%s",
            training_id, best_round, best_accuracy, converged,
        )

    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
        temperature: float = 1.0,
        training_id: Optional[int] = None,
    ) -> None:
        data = _serialize(state_dict, vocab, temperature, checkpoint_round=round_num)
        sha256 = hashlib.sha256(data).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            if training_id is not None:
                conn.execute(
                    "INSERT INTO fl_checkpoints "
                    "(round, accuracy, loss, model_bytes, sha256, vocab_size, training_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(training_id) DO UPDATE SET "
                    "round=excluded.round, accuracy=excluded.accuracy, loss=excluded.loss, "
                    "model_bytes=excluded.model_bytes, sha256=excluded.sha256, "
                    "vocab_size=excluded.vocab_size",
                    (round_num, accuracy, loss, data, sha256, len(vocab), training_id, created_at),
                )
            else:
                conn.execute(
                    "INSERT INTO fl_checkpoints "
                    "(round, accuracy, loss, model_bytes, sha256, vocab_size, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (round_num, accuracy, loss, data, sha256, len(vocab), created_at),
                )
        logger.info(
            "checkpoint_saved_sqlite round=%d accuracy=%.4f training_id=%s sha256=%s",
            round_num, accuracy, training_id, sha256[:12],
        )

    def load_latest(self) -> Optional[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT model_bytes, sha256 FROM fl_checkpoints ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data, stored_sha256 = bytes(row[0]), row[1]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        return _deserialize(data)

    def load_best(self, training_id: Optional[int] = None) -> Optional[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            if training_id is not None:
                row = conn.execute(
                    "SELECT model_bytes, sha256, round, accuracy FROM fl_checkpoints "
                    "WHERE training_id=? LIMIT 1",
                    (training_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT model_bytes, sha256, round, accuracy FROM fl_checkpoints "
                    "ORDER BY accuracy DESC LIMIT 1"
                ).fetchone()
        if row is None:
            return None
        data, stored_sha256, round_num, accuracy = bytes(row[0]), row[1], row[2], row[3]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        logger.info("checkpoint_best_loaded_sqlite round=%d accuracy=%.4f training_id=%s",
                    round_num, accuracy, training_id)
        return _deserialize(data)


class PostgreSQLCheckpointStore(CheckpointStore):
    """Checkpoint store em PostgreSQL puro — para produção/homologação."""

    def __init__(self, db_url: str) -> None:
        import sqlalchemy as sa
        # Schema gerenciado pelo Alembic (migration 011_fl_trainings) — sem DDL aqui.
        self._engine = sa.create_engine(db_url, pool_pre_ping=True)
        logger.info("postgres_checkpoint_store_ready")

    def register_training(
        self,
        algorithm: str = "FedAvg",
        log_file: str = "",
        n_rounds_max: int = 120,
    ) -> int:
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            row = conn.execute(
                sa.text(
                    "INSERT INTO metrics.fl_trainings (algorithm, log_file, n_rounds_max) "
                    "VALUES (:algorithm, :log_file, :n_rounds_max) RETURNING id"
                ),
                {"algorithm": algorithm, "log_file": log_file, "n_rounds_max": n_rounds_max},
            ).fetchone()
        training_id = row[0]
        logger.info("training_registered_postgres id=%d algorithm=%s", training_id, algorithm)
        return training_id

    def complete_training(
        self,
        training_id: int,
        n_rounds_done: int,
        best_round: int,
        best_accuracy: float,
        converged: bool,
    ) -> None:
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE metrics.fl_trainings SET status='completed', completed_at=NOW(), "
                    "n_rounds_done=:n_rounds_done, best_round=:best_round, "
                    "best_accuracy=:best_accuracy, converged=:converged "
                    "WHERE id=:training_id"
                ),
                {
                    "training_id":   training_id,
                    "n_rounds_done": n_rounds_done,
                    "best_round":    best_round,
                    "best_accuracy": best_accuracy,
                    "converged":     converged,
                },
            )
        logger.info(
            "training_completed_postgres id=%d best_round=%d best_accuracy=%.4f converged=%s",
            training_id, best_round, best_accuracy, converged,
        )

    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
        temperature: float = 1.0,
        training_id: Optional[int] = None,
    ) -> None:
        import sqlalchemy as sa
        data = _serialize(state_dict, vocab, temperature, checkpoint_round=round_num)
        sha256 = hashlib.sha256(data).hexdigest()
        if training_id is not None:
            sql = sa.text(
                "INSERT INTO metrics.fl_checkpoints "
                "(round, accuracy, loss, model_bytes, sha256, vocab_size, training_id) "
                "VALUES (:round, :accuracy, :loss, :model_bytes, :sha256, :vocab_size, :training_id) "
                "ON CONFLICT (training_id) WHERE training_id IS NOT NULL DO UPDATE SET "
                "round=EXCLUDED.round, accuracy=EXCLUDED.accuracy, loss=EXCLUDED.loss, "
                "model_bytes=EXCLUDED.model_bytes, sha256=EXCLUDED.sha256, "
                "vocab_size=EXCLUDED.vocab_size"
            )
        else:
            sql = sa.text(
                "INSERT INTO metrics.fl_checkpoints "
                "(round, accuracy, loss, model_bytes, sha256, vocab_size) "
                "VALUES (:round, :accuracy, :loss, :model_bytes, :sha256, :vocab_size)"
            )
        params = {
            "round": round_num, "accuracy": accuracy, "loss": loss,
            "model_bytes": data, "sha256": sha256, "vocab_size": len(vocab),
        }
        if training_id is not None:
            params["training_id"] = training_id
        with self._engine.begin() as conn:
            conn.execute(sql, params)
        logger.info(
            "checkpoint_saved_postgres round=%d accuracy=%.4f training_id=%s sha256=%s",
            round_num, accuracy, training_id, sha256[:12],
        )

    def load_latest(self) -> Optional[Dict]:
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT model_bytes, sha256 FROM metrics.fl_checkpoints "
                    "ORDER BY id DESC LIMIT 1"
                )
            ).fetchone()
        if row is None:
            return None
        data, stored_sha256 = bytes(row[0]), row[1]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        return _deserialize(data)

    def load_best(self, training_id: Optional[int] = None) -> Optional[Dict]:
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            if training_id is not None:
                row = conn.execute(
                    sa.text(
                        "SELECT model_bytes, sha256, round, accuracy "
                        "FROM metrics.fl_checkpoints WHERE training_id=:training_id LIMIT 1"
                    ),
                    {"training_id": training_id},
                ).fetchone()
            else:
                row = conn.execute(
                    sa.text(
                        "SELECT model_bytes, sha256, round, accuracy "
                        "FROM metrics.fl_checkpoints ORDER BY accuracy DESC LIMIT 1"
                    )
                ).fetchone()
        if row is None:
            return None
        data, stored_sha256, round_num, accuracy = bytes(row[0]), row[1], row[2], row[3]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        logger.info("checkpoint_best_loaded_postgres round=%d accuracy=%.4f training_id=%s",
                    round_num, accuracy, training_id)
        return _deserialize(data)


def get_checkpoint_store(db_url: str = "") -> CheckpointStore:
    """
    Retorna o store adequado ao ambiente:
      FL_DB_URL configurado  → PostgreSQLCheckpointStore (produção/homologação)
      FL_DB_URL vazio        → SQLiteCheckpointStore     (experimentos)
    """
    if db_url:
        return PostgreSQLCheckpointStore(db_url)
    return SQLiteCheckpointStore()
