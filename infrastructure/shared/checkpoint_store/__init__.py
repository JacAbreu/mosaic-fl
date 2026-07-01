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

Submódulos:
  serialization.py  — _model_version, _serialize, _deserialize, DDL SQLite
  base.py            — CheckpointStore (interface ABC)
  sqlite_store.py     — SQLiteCheckpointStore
  postgres_store.py    — PostgreSQLCheckpointStore
"""
from .base import CheckpointStore
from .postgres_store import PostgreSQLCheckpointStore
from .sqlite_store import SQLiteCheckpointStore


def get_checkpoint_store(db_url: str = "") -> CheckpointStore:
    """
    Retorna o store adequado ao ambiente:
      FL_DB_URL configurado  → PostgreSQLCheckpointStore (produção/homologação)
      FL_DB_URL vazio        → SQLiteCheckpointStore     (experimentos)
    """
    if db_url:
        return PostgreSQLCheckpointStore(db_url)
    return SQLiteCheckpointStore()


__all__ = [
    "CheckpointStore",
    "SQLiteCheckpointStore",
    "PostgreSQLCheckpointStore",
    "get_checkpoint_store",
]
