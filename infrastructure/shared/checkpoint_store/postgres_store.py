"""postgres_store.py — CheckpointStore em PostgreSQL puro, para produção/homologação."""
import hashlib
import logging
from collections import OrderedDict
from typing import Dict, Optional

from .base import CheckpointStore
from .serialization import _deserialize, _serialize

logger = logging.getLogger(__name__)


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
        partition_mode: str = "natural",
        run_classification: str = "ajuste",
    ) -> int:
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            row = conn.execute(
                sa.text(
                    "INSERT INTO metrics.fl_trainings "
                    "(algorithm, log_file, n_rounds_max, checkpoint_criterion, partition_mode, run_classification) "
                    "VALUES (:algorithm, :log_file, :n_rounds_max, :checkpoint_criterion, :partition_mode, :run_classification) "
                    "RETURNING id"
                ),
                {
                    "algorithm":            algorithm,
                    "log_file":             log_file,
                    "n_rounds_max":         n_rounds_max,
                    "checkpoint_criterion": checkpoint_criterion,
                    "partition_mode":       partition_mode,
                    "run_classification":   run_classification,
                },
            ).fetchone()
        training_id = row[0]
        logger.info(
            "training_registered_postgres id=%d algorithm=%s criterion=%s partition_mode=%s run_classification=%s",
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
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE metrics.fl_trainings SET status='completed', completed_at=NOW(), "
                    "n_rounds_done=:n_rounds_done, best_round=:best_round, "
                    "best_accuracy=:best_accuracy, converged=:converged, "
                    "total_duration_s=:total_duration_s, peak_ram_mb=:peak_ram_mb, "
                    "avg_cpu_pct=:avg_cpu_pct, gpu_avg_power_w=:gpu_avg_power_w, "
                    "gpu_peak_power_w=:gpu_peak_power_w, gpu_energy_wh=:gpu_energy_wh "
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
                    "gpu_avg_power_w":  gpu_avg_power_w,
                    "gpu_peak_power_w": gpu_peak_power_w,
                    "gpu_energy_wh":    gpu_energy_wh,
                },
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
        import sqlalchemy as sa
        with self._engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE metrics.fl_trainings SET macro_auc=:macro_auc, "
                    "macro_f1=:macro_f1, ece=:ece, ece_pre=:ece_pre, "
                    "dp_noise_multiplier=:dp_noise_multiplier, dp_max_grad_norm=:dp_max_grad_norm, "
                    "dp_epsilon_simple=:dp_epsilon_simple, dp_epsilon_rdp=:dp_epsilon_rdp "
                    "WHERE id=:training_id"
                ),
                {
                    "training_id":         training_id,
                    "macro_auc":           macro_auc,
                    "macro_f1":            macro_f1,
                    "ece":                 ece,
                    "ece_pre":             ece_pre,
                    "dp_noise_multiplier": dp_noise_multiplier,
                    "dp_max_grad_norm":    dp_max_grad_norm,
                    "dp_epsilon_simple":   dp_epsilon_simple,
                    "dp_epsilon_rdp":      dp_epsilon_rdp,
                },
            )
        logger.info(
            "training_evaluation_metrics_saved id=%d macro_auc=%s macro_f1=%s ece=%s ece_pre=%s "
            "dp_sigma=%s dp_clip=%s dp_eps_simple=%s dp_eps_rdp=%s",
            training_id, macro_auc, macro_f1, ece, ece_pre,
            dp_noise_multiplier, dp_max_grad_norm, dp_epsilon_simple, dp_epsilon_rdp,
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
