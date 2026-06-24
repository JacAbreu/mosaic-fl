"""
MetricsStore — persistência de métricas de avaliação do modelo federado.

Dois backends com interface idêntica (mesmo padrão do CheckpointStore):
  SQLiteMetricsStore      — experimentos locais (sem servidor de banco)
  PostgreSQLMetricsStore  — produção/homologação

Seleção automática via get_metrics_store():
  FL_DB_URL configurado → PostgreSQLMetricsStore
  FL_DB_URL vazio       → SQLiteMetricsStore

O que é persistido por round:
  - Métricas globais do modelo FL (accuracy, loss, macro_auc, macro_f1, ece)
  - Métricas por classe (auc por classe, f1 por classe) em JSON
  - Métricas de recuperação do RAG (precision@k, por classe)
  - SHA-256 do checkpoint correspondente (rastreabilidade)
  - Fonte dos dados ('synthetic', 'fapesp', 'real')
"""
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS fl_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    round                   INTEGER NOT NULL,
    checkpoint_sha256       TEXT,
    accuracy                REAL,
    loss                    REAL,
    macro_auc               REAL,
    macro_f1                REAL,
    ece                     REAL,
    per_class_auc           TEXT,
    per_class_f1            TEXT,
    rag_precision_at_k      REAL,
    rag_k                   INTEGER,
    rag_per_class_precision TEXT,
    data_source             TEXT NOT NULL DEFAULT 'synthetic',
    evaluated_at            TEXT NOT NULL
)
"""

_CREATE_POSTGRES = """
CREATE TABLE IF NOT EXISTS metrics.fl_metrics (
    id                      SERIAL PRIMARY KEY,
    round                   INTEGER     NOT NULL,
    checkpoint_sha256       TEXT,
    accuracy                REAL,
    loss                    REAL,
    macro_auc               REAL,
    macro_f1                REAL,
    ece                     REAL,
    per_class_auc           JSONB,
    per_class_f1            JSONB,
    rag_precision_at_k      REAL,
    rag_k                   INTEGER,
    rag_per_class_precision JSONB,
    data_source             TEXT        NOT NULL DEFAULT 'synthetic',
    evaluated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class MetricsStore(ABC):
    """Interface para persistência de métricas de avaliação federada."""

    @abstractmethod
    def save(
        self,
        round_num: int,
        metrics: Dict,
        checkpoint_sha256: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        """
        Persiste métricas de avaliação de um round.

        metrics pode conter:
          accuracy, loss, macro_auc, macro_f1, ece           — métricas globais do modelo
          per_class_auc, per_class_f1                        — dicts {classe: valor}
          rag_precision_at_k, rag_k, rag_per_class_precision — métricas do RAG
        """

    @abstractmethod
    def load_history(self, last_n: Optional[int] = None) -> list:
        """Retorna histórico de métricas, opcionalmente limitado aos últimos N rounds."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna as métricas do round mais recente, ou None."""


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


def _row_to_dict(row: Dict) -> Dict:
    """Desserializa campos JSON armazenados como string."""
    for field in ("per_class_auc", "per_class_f1", "rag_per_class_precision"):
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return row


def get_metrics_store(db_url: str = "") -> MetricsStore:
    """
    Retorna o store adequado ao ambiente:
      FL_DB_URL configurado  → PostgreSQLMetricsStore (produção/homologação)
      FL_DB_URL vazio        → SQLiteMetricsStore     (experimentos)
    """
    if db_url:
        return PostgreSQLMetricsStore(db_url)
    return SQLiteMetricsStore()
