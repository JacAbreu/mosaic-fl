"""postgres_store.py — MetricsStore em PostgreSQL, para produção/homologação."""
import json
import logging
from typing import Dict, Optional

from .base import MetricsStore
from .serialization import _CREATE_POSTGRES, _row_to_dict

logger = logging.getLogger(__name__)


class PostgreSQLMetricsStore(MetricsStore):
    """Métricas em PostgreSQL — para produção/homologação."""

    def __init__(self, db_url: str) -> None:
        import sqlalchemy as sa
        self._engine = sa.create_engine(db_url, pool_pre_ping=True)
        with self._engine.begin() as conn:
            conn.execute(sa.text(_CREATE_POSTGRES))
        logger.info("postgres_metrics_store_ready")

    def save(
        self,
        round_num: int,
        metrics: Dict,
        checkpoint_sha256: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        import sqlalchemy as sa
        per_class_auc = metrics.get("per_class_auc")
        per_class_f1  = metrics.get("per_class_f1")
        rag_per_class = metrics.get("rag_per_class_precision")

        with self._engine.begin() as conn:
            conn.execute(
                sa.text("""
                    INSERT INTO metrics.fl_metrics
                    (round, checkpoint_sha256, accuracy, loss, macro_auc, macro_f1, ece,
                     per_class_auc, per_class_f1,
                     rag_precision_at_k, rag_k, rag_per_class_precision,
                     data_source)
                    VALUES
                    (:round, :sha256, :accuracy, :loss, :macro_auc, :macro_f1, :ece,
                     :per_class_auc, :per_class_f1,
                     :rag_pk, :rag_k, :rag_per_class,
                     :data_source)
                """),
                {
                    "round":         round_num,
                    "sha256":        checkpoint_sha256,
                    "accuracy":      metrics.get("accuracy"),
                    "loss":          metrics.get("loss"),
                    "macro_auc":     metrics.get("macro_auc"),
                    "macro_f1":      metrics.get("macro_f1"),
                    "ece":           metrics.get("ece"),
                    "per_class_auc": json.dumps(per_class_auc) if per_class_auc else None,
                    "per_class_f1":  json.dumps(per_class_f1)  if per_class_f1  else None,
                    "rag_pk":        metrics.get("rag_precision_at_k"),
                    "rag_k":         metrics.get("rag_k"),
                    "rag_per_class": json.dumps(rag_per_class) if rag_per_class else None,
                    "data_source":   data_source,
                },
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
        import sqlalchemy as sa
        query = "SELECT * FROM metrics.fl_metrics ORDER BY id DESC"
        if last_n:
            query += f" LIMIT {int(last_n)}"
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(query)).mappings().all()
        return [_row_to_dict(dict(r)) for r in rows]

    def load_latest(self) -> Optional[Dict]:
        import sqlalchemy as sa
        with self._engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT * FROM metrics.fl_metrics ORDER BY id DESC LIMIT 1"
                )
            ).mappings().fetchone()
        return _row_to_dict(dict(row)) if row else None
