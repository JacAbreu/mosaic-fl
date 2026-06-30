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
        checkpoint_criterion: str = "f1_macro",
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
        total_duration_s: float = 0.0,
        peak_ram_mb: float = 0.0,
        avg_cpu_pct: float = 0.0,
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
        evaluation_json: Optional[Dict] = None,
    ) -> None:
        """UPSERT do checkpoint: 1 linha por training_id (substitui quando Acc melhora)."""

    @abstractmethod
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
        """Persiste accuracy, loss, f1_macro, τ_eff, per_class_f1 e round_duration_s por rodada.
        tau_effs é None por elemento quando o algoritmo é FedAvg."""

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
        checkpoint_criterion: str = "f1_macro",
    ) -> int:
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            row = conn.execute(
                sa.text(
                    "INSERT INTO metrics.fl_trainings (algorithm, log_file, n_rounds_max, checkpoint_criterion) "
                    "VALUES (:algorithm, :log_file, :n_rounds_max, :checkpoint_criterion) RETURNING id"
                ),
                {
                    "algorithm":            algorithm,
                    "log_file":             log_file,
                    "n_rounds_max":         n_rounds_max,
                    "checkpoint_criterion": checkpoint_criterion,
                },
            ).fetchone()
        training_id = row[0]
        logger.info("training_registered_postgres id=%d algorithm=%s criterion=%s",
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
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE metrics.fl_trainings SET status='completed', completed_at=NOW(), "
                    "n_rounds_done=:n_rounds_done, best_round=:best_round, "
                    "best_accuracy=:best_accuracy, converged=:converged, "
                    "total_duration_s=:total_duration_s, peak_ram_mb=:peak_ram_mb, "
                    "avg_cpu_pct=:avg_cpu_pct "
                    "WHERE id=:training_id"
                ),
                {
                    "training_id":      training_id,
                    "n_rounds_done":    n_rounds_done,
                    "best_round":       best_round,
                    "best_accuracy":    best_accuracy,
                    "converged":        converged,
                    "total_duration_s": total_duration_s,
                    "peak_ram_mb":      peak_ram_mb,
                    "avg_cpu_pct":      avg_cpu_pct,
                },
            )
        logger.info(
            "training_completed_postgres id=%d best_round=%d best_accuracy=%.4f converged=%s "
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
        import json as _json
        import sqlalchemy as sa
        data = _serialize(state_dict, vocab, temperature, checkpoint_round=round_num)
        sha256 = hashlib.sha256(data).hexdigest()
        eval_json_str = _json.dumps(evaluation_json, ensure_ascii=False) if evaluation_json else None
        if training_id is not None:
            sql = sa.text(
                "INSERT INTO metrics.fl_checkpoints "
                "(round, accuracy, loss, model_bytes, sha256, vocab_size, training_id, evaluation_json) "
                "VALUES (:round, :accuracy, :loss, :model_bytes, :sha256, :vocab_size, :training_id, cast(:evaluation_json as jsonb)) "
                "ON CONFLICT (training_id) WHERE training_id IS NOT NULL DO UPDATE SET "
                "round=EXCLUDED.round, accuracy=EXCLUDED.accuracy, loss=EXCLUDED.loss, "
                "model_bytes=EXCLUDED.model_bytes, sha256=EXCLUDED.sha256, "
                "vocab_size=EXCLUDED.vocab_size, evaluation_json=EXCLUDED.evaluation_json"
            )
        else:
            sql = sa.text(
                "INSERT INTO metrics.fl_checkpoints "
                "(round, accuracy, loss, model_bytes, sha256, vocab_size, evaluation_json) "
                "VALUES (:round, :accuracy, :loss, :model_bytes, :sha256, :vocab_size, cast(:evaluation_json as jsonb))"
            )
        params = {
            "round": round_num, "accuracy": accuracy, "loss": loss,
            "model_bytes": data, "sha256": sha256, "vocab_size": len(vocab),
            "evaluation_json": eval_json_str,
        }
        if training_id is not None:
            params["training_id"] = training_id
        with self._engine.begin() as conn:
            conn.execute(sql, params)
        logger.info(
            "checkpoint_saved_postgres round=%d accuracy=%.4f training_id=%s sha256=%s evaluation_json=%s",
            round_num, accuracy, training_id, sha256[:12],
            "saved" if evaluation_json else "null",
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
        import json as _json
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            if training_id is not None:
                row = conn.execute(
                    sa.text(
                        "SELECT model_bytes, sha256, round, accuracy, evaluation_json "
                        "FROM metrics.fl_checkpoints WHERE training_id=:training_id LIMIT 1"
                    ),
                    {"training_id": training_id},
                ).fetchone()
            else:
                row = conn.execute(
                    sa.text(
                        "SELECT model_bytes, sha256, round, accuracy, evaluation_json "
                        "FROM metrics.fl_checkpoints ORDER BY accuracy DESC LIMIT 1"
                    )
                ).fetchone()
        if row is None:
            return None
        data, stored_sha256, round_num, accuracy, eval_json = (
            bytes(row[0]), row[1], row[2], row[3], row[4]
        )
        if hashlib.sha256(data).hexdigest() != stored_sha256:
            logger.error("checkpoint_integrity_error sha256 mismatch — checkpoint descartado")
            return None
        logger.info("checkpoint_best_loaded_postgres round=%d accuracy=%.4f training_id=%s",
                    round_num, accuracy, training_id)
        result = _deserialize(data)
        if eval_json is not None:
            result["evaluation_json"] = eval_json if isinstance(eval_json, dict) else _json.loads(eval_json)
        return result

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
        import json as _json
        import sqlalchemy as sa
        _tau  = tau_effs        if tau_effs        is not None else [None] * len(rounds)
        _f1   = f1_macros       if f1_macros       is not None else [None] * len(rounds)
        _pcf1 = per_class_f1s   if per_class_f1s   is not None else [None] * len(rounds)
        _dur  = round_durations if round_durations  is not None else [None] * len(rounds)
        rows = [
            {
                "training_id":     training_id,
                "round":           r,
                "accuracy":        float(a),
                "loss":            float(l),
                "tau_eff":         float(t)   if t   is not None else None,
                "f1_macro":        float(f)   if f   is not None else None,
                "per_class_f1":    _json.dumps(pc) if pc is not None else None,
                "round_duration_s": float(d)  if d   is not None else None,
            }
            for r, a, l, t, f, pc, d in zip(rounds, accuracies, losses, _tau, _f1, _pcf1, _dur)
        ]
        if not rows:
            return
        sql = sa.text(
            "INSERT INTO metrics.fl_round_history "
            "(training_id, round, accuracy, loss, tau_eff, f1_macro, per_class_f1, round_duration_s) "
            "VALUES (:training_id, :round, :accuracy, :loss, :tau_eff, :f1_macro, "
            "cast(:per_class_f1 as jsonb), :round_duration_s) "
            "ON CONFLICT (training_id, round) DO UPDATE SET "
            "accuracy=EXCLUDED.accuracy, loss=EXCLUDED.loss, tau_eff=EXCLUDED.tau_eff, "
            "f1_macro=EXCLUDED.f1_macro, per_class_f1=EXCLUDED.per_class_f1, "
            "round_duration_s=EXCLUDED.round_duration_s"
        )
        with self._engine.begin() as conn:
            conn.execute(sql, rows)
        logger.info(
            "round_history_saved training_id=%d rounds=%d tau=%s f1=%s per_class=%s dur=%s",
            training_id, len(rows),
            tau_effs is not None, f1_macros is not None,
            per_class_f1s is not None, round_durations is not None,
        )


def get_checkpoint_store(db_url: str = "") -> CheckpointStore:
    """
    Retorna o store adequado ao ambiente:
      FL_DB_URL configurado  → PostgreSQLCheckpointStore (produção/homologação)
      FL_DB_URL vazio        → SQLiteCheckpointStore     (experimentos)
    """
    if db_url:
        return PostgreSQLCheckpointStore(db_url)
    return SQLiteCheckpointStore()
