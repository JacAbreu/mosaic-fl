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

Submódulos:
  serialization.py  — DDL das tabelas + _row_to_dict
  base.py             — MetricsStore (interface ABC)
  sqlite_store.py       — SQLiteMetricsStore
  postgres_store.py       — PostgreSQLMetricsStore
"""
from .base import MetricsStore
from .postgres_store import PostgreSQLMetricsStore
from .sqlite_store import SQLiteMetricsStore


def get_metrics_store(db_url: str = "") -> MetricsStore:
    """
    Retorna o store adequado ao ambiente:
      FL_DB_URL configurado  → PostgreSQLMetricsStore (produção/homologação)
      FL_DB_URL vazio        → SQLiteMetricsStore     (experimentos)
    """
    if db_url:
        return PostgreSQLMetricsStore(db_url)
    return SQLiteMetricsStore()


__all__ = [
    "MetricsStore",
    "SQLiteMetricsStore",
    "PostgreSQLMetricsStore",
    "get_metrics_store",
]
