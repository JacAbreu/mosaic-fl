"""
CheckpointStore — persistência de pesos do modelo federado.

Dois backends com interface idêntica:
  SQLiteCheckpointStore   — experimentos locais (sem servidor de banco)
  PostgreSQLCheckpointStore — produção/homologação (PostgreSQL puro, sem pgvector)

Migração futura para S3: substitui apenas os bytes (store → S3 key no banco).

Seleção automática via get_checkpoint_store():
  FL_DB_URL configurado → PostgreSQLCheckpointStore
  FL_DB_URL vazio       → SQLiteCheckpointStore
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
    created_at  TEXT    NOT NULL
)
"""

_CREATE_POSTGRES = """
CREATE TABLE IF NOT EXISTS metrics.fl_checkpoints (
    id          SERIAL PRIMARY KEY,
    round       INTEGER     NOT NULL,
    accuracy    REAL        NOT NULL DEFAULT 0.0,
    loss        REAL        NOT NULL DEFAULT 0.0,
    model_bytes BYTEA       NOT NULL,
    sha256      TEXT        NOT NULL,
    vocab_size  INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _serialize(state_dict: OrderedDict, vocab: Dict[str, int]) -> bytes:
    buf = io.BytesIO()
    torch.save({"model_state": state_dict, "vocab": vocab}, buf)
    return buf.getvalue()


def _deserialize(data: bytes) -> Dict:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


class CheckpointStore(ABC):
    """Interface para persistência de checkpoints federados."""

    @abstractmethod
    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
    ) -> None:
        """Serializa state_dict + vocab e persiste com metadados da rodada."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna {'model_state': OrderedDict, 'vocab': dict} do checkpoint mais recente, ou None."""


class SQLiteCheckpointStore(CheckpointStore):
    """Checkpoint store em SQLite — para experimentos locais."""

    def __init__(self, db_path: str = "checkpoints/experiment.db") -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQLITE)
        logger.info("sqlite_checkpoint_store_ready path=%s", db_path)

    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
    ) -> None:
        data = _serialize(state_dict, vocab)
        sha256 = hashlib.sha256(data).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO fl_checkpoints "
                "(round, accuracy, loss, model_bytes, sha256, vocab_size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (round_num, accuracy, loss, data, sha256, len(vocab), created_at),
            )
        logger.info(
            "checkpoint_saved_sqlite round=%d accuracy=%.4f vocab_size=%d sha256=%s",
            round_num, accuracy, len(vocab), sha256[:12],
        )

    def load_latest(self) -> Optional[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT model_bytes, sha256 FROM fl_checkpoints ORDER BY round DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data, stored_sha256 = bytes(row[0]), row[1]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        return _deserialize(data)


class PostgreSQLCheckpointStore(CheckpointStore):
    """Checkpoint store em PostgreSQL puro — para produção/homologação."""

    def __init__(self, db_url: str) -> None:
        import sqlalchemy as sa
        self._engine = sa.create_engine(db_url, pool_pre_ping=True)
        with self._engine.begin() as conn:
            conn.execute(sa.text(_CREATE_POSTGRES))
        logger.info("postgres_checkpoint_store_ready")

    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
    ) -> None:
        import sqlalchemy as sa
        data = _serialize(state_dict, vocab)
        sha256 = hashlib.sha256(data).hexdigest()
        with self._engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO metrics.fl_checkpoints "
                    "(round, accuracy, loss, model_bytes, sha256, vocab_size) "
                    "VALUES (:round, :accuracy, :loss, :model_bytes, :sha256, :vocab_size)"
                ),
                {
                    "round": round_num,
                    "accuracy": accuracy,
                    "loss": loss,
                    "model_bytes": data,
                    "sha256": sha256,
                    "vocab_size": len(vocab),
                },
            )
        logger.info(
            "checkpoint_saved_postgres round=%d accuracy=%.4f vocab_size=%d sha256=%s",
            round_num, accuracy, len(vocab), sha256[:12],
        )

    def load_latest(self) -> Optional[Dict]:
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT model_bytes, sha256 FROM metrics.fl_checkpoints "
                    "ORDER BY round DESC LIMIT 1"
                )
            ).fetchone()
        if row is None:
            return None
        data, stored_sha256 = bytes(row[0]), row[1]
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
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
