"""sqlite_store.py — CheckpointStore em SQLite, para experimentos locais (sem servidor de banco)."""
import hashlib
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

from .base import CheckpointStore
from .serialization import _CREATE_SQLITE, _CREATE_SQLITE_TRAININGS, _deserialize, _serialize

logger = logging.getLogger(__name__)


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
        checkpoint_criterion: str = "f1_macro",
    ) -> int:
        started_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO fl_trainings (algorithm, log_file, n_rounds_max, started_at) "
                "VALUES (?, ?, ?, ?)",
                (algorithm, log_file, n_rounds_max, started_at),
            )
            training_id = cur.lastrowid
        logger.info("training_registered_sqlite id=%d algorithm=%s criterion=%s",
                    training_id, algorithm, checkpoint_criterion)
        return training_id

    def complete_training(
        self,
        training_id: int,
        n_rounds_done: int,
        best_round: int,
        best_accuracy: float,
        converged: bool,
        total_duration_s: float = 0.0,
        peak_ram_mb: float = 0.0,
        avg_cpu_pct: float = 0.0,
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
            "training_completed_sqlite id=%d best_round=%d best_accuracy=%.4f converged=%s "
            "duration=%.1fs peak_ram=%.0fMB avg_cpu=%.1f%%",
            training_id, best_round, best_accuracy, converged,
            total_duration_s, peak_ram_mb, avg_cpu_pct,
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
        evaluation_json: Optional[Dict] = None,
    ) -> None:
        # evaluation_json não é persistido no SQLite (backend de simulação).
        # Produção usa PostgreSQLCheckpointStore, que persiste via migration 012.
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

    def save_round_history(
        self,
        training_id: int,
        rounds: list,
        accuracies: list,
        losses: list,
        tau_effs: Optional[list] = None,
        f1_macros: Optional[list] = None,
        per_class_f1s: Optional[list] = None,
        round_durations: Optional[list] = None,
    ) -> None:
        # SQLite é o backend de simulação — fl_round_history existe apenas no PostgreSQL (migration 013).
        logger.debug("save_round_history: no-op no SQLiteCheckpointStore (training_id=%d)", training_id)
