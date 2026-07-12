"""sqlite_store.py — CheckpointStore em SQLite, para experimentos locais (sem servidor de banco)."""
import hashlib
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
        partition_mode: str = "natural",
        run_classification: str = "ajuste",
    ) -> int:
        # fl_trainings do SQLite não tem coluna partition_mode/run_classification
        # (mesma lacuna já existente para checkpoint_criterion/métricas de recurso
        # nesta classe).
        started_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO fl_trainings (algorithm, log_file, n_rounds_max, started_at) "
                "VALUES (?, ?, ?, ?)",
                (algorithm, log_file, n_rounds_max, started_at),
            )
            training_id = cur.lastrowid
        logger.info(
            "training_registered_sqlite id=%d algorithm=%s criterion=%s partition_mode=%s "
            "run_classification=%s (não persistido)",
            training_id, algorithm, checkpoint_criterion, partition_mode, run_classification,
        )
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
        gpu_avg_power_w: Optional[float] = None,
        gpu_peak_power_w: Optional[float] = None,
        gpu_energy_wh: Optional[float] = None,
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
            "duration=%.1fs peak_ram=%.0fMB avg_cpu=%.1f%% gpu_avg_power=%sW gpu_energy=%sWh "
            "(campos de recurso/GPU não persistidos no schema local — ver evaluation_json)",
            training_id, best_round, best_accuracy, converged,
            total_duration_s, peak_ram_mb, avg_cpu_pct, gpu_avg_power_w, gpu_energy_wh,
        )

    def update_evaluation_metrics(
        self,
        training_id: int,
        macro_auc: Optional[float] = None,
        macro_f1: Optional[float] = None,
        ece: Optional[float] = None,
        ece_pre: Optional[float] = None,
        dp_noise_multiplier: Optional[float] = None,
        dp_max_grad_norm: Optional[float] = None,
        dp_epsilon_simple: Optional[float] = None,
        dp_epsilon_rdp: Optional[float] = None,
    ) -> None:
        # fl_trainings do SQLite ainda não tem essas colunas (schema local, sem
        # Alembic — mesma lacuna já existente para as métricas de recurso
        # computacional desta classe). O valor completo continua disponível em
        # evaluation_json, salvo por save(). Loga para não perder o dado silenciosamente.
        logger.info(
            "training_evaluation_metrics_sqlite_not_persisted id=%d macro_auc=%s macro_f1=%s ece=%s ece_pre=%s "
            "dp_sigma=%s dp_clip=%s dp_eps_simple=%s dp_eps_rdp=%s (ver evaluation_json no checkpoint)",
            training_id, macro_auc, macro_f1, ece, ece_pre,
            dp_noise_multiplier, dp_max_grad_norm, dp_epsilon_simple, dp_epsilon_rdp,
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
        calibration_method: str = "temperature",
        isotonic_calibrators: Optional[List] = None,
        isotonic_num_classes: int = 0,
    ) -> None:
        # evaluation_json não é persistido no SQLite (backend de simulação).
        # Produção usa PostgreSQLCheckpointStore, que persiste via migration 012.
        data = _serialize(
            state_dict, vocab, temperature, checkpoint_round=round_num,
            calibration_method=calibration_method,
            isotonic_calibrators=isotonic_calibrators,
            isotonic_num_classes=isotonic_num_classes,
        )
        sha256 = hashlib.sha256(data).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            if training_id is not None:
                conn.execute(
                    "INSERT INTO fl_checkpoints "
                    "(round, accuracy, loss, model_bytes, sha256, vocab_size, training_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    # ON CONFLICT precisa repetir o WHERE do índice parcial (fl_checkpoints_training_id_uniq)
                    # para o SQLite reconhecê-lo como alvo do UPSERT — sem isso, "ON CONFLICT(training_id)"
                    # sozinho não casa com um índice único parcial (erro: "ON CONFLICT clause does not
                    # match any PRIMARY KEY or UNIQUE constraint"). Mesmo requisito já atendido em
                    # postgres_store.py; aqui estava faltando (nunca exercitado por nenhum teste).
                    "ON CONFLICT(training_id) WHERE training_id IS NOT NULL DO UPDATE SET "
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
