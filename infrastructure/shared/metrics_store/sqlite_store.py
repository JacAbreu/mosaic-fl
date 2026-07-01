"""sqlite_store.py — MetricsStore em SQLite, para experimentos locais (sem servidor de banco)."""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Optional

from .base import MetricsStore
from .serialization import _CREATE_SQLITE, _row_to_dict

logger = logging.getLogger(__name__)


class SQLiteMetricsStore(MetricsStore):
    """Métricas em SQLite — para experimentos locais."""

    def __init__(self, db_path: str = "checkpoints/experiment.db") -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQLITE)
        logger.info("sqlite_metrics_store_ready path=%s", db_path)

    def save(
        self,
        round_num: int,
        metrics: Dict,
        checkpoint_sha256: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        per_class_auc = metrics.get("per_class_auc")
        per_class_f1  = metrics.get("per_class_f1")
        rag_per_class = metrics.get("rag_per_class_precision")

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO fl_metrics
                   (round, checkpoint_sha256, accuracy, loss, macro_auc, macro_f1, ece,
                    per_class_auc, per_class_f1,
                    rag_precision_at_k, rag_k, rag_per_class_precision,
                    data_source, evaluated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    round_num,
                    checkpoint_sha256,
                    metrics.get("accuracy"),
                    metrics.get("loss"),
                    metrics.get("macro_auc"),
                    metrics.get("macro_f1"),
                    metrics.get("ece"),
                    json.dumps(per_class_auc)  if per_class_auc  is not None else None,
                    json.dumps(per_class_f1)   if per_class_f1   is not None else None,
                    metrics.get("rag_precision_at_k"),
                    metrics.get("rag_k"),
                    json.dumps(rag_per_class)  if rag_per_class  is not None else None,
                    data_source,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.info(
            "metrics_saved round=%d acc=%.4f auc=%s ece=%s rag_p@k=%s source=%s",
            round_num,
            metrics.get("accuracy") or 0.0,
            f"{metrics['macro_auc']:.4f}" if metrics.get("macro_auc") else "n/a",
            f"{metrics['ece']:.4f}"       if metrics.get("ece")       else "n/a",
            f"{metrics['rag_precision_at_k']:.4f}" if metrics.get("rag_precision_at_k") else "n/a",
            data_source,
        )

    def load_history(self, last_n: Optional[int] = None) -> list:
        query = "SELECT * FROM fl_metrics ORDER BY id DESC"
        if last_n:
            query += f" LIMIT {int(last_n)}"
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
        return [_row_to_dict(dict(r)) for r in rows]

    def load_latest(self) -> Optional[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM fl_metrics ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return _row_to_dict(dict(row)) if row else None
