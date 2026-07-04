"""fl_checkpoints — cria a tabela base, nunca antes registrada em migration alguma

Encontrado em 2026-07-04, ao inicializar um segundo banco (mosaicfl-db-bpsp,
cenário desktop+notebook) do zero: `alembic upgrade head` falhava na migration
011 (`ALTER TABLE metrics.fl_checkpoints ADD COLUMN ...`) porque a tabela nunca
existiu em nenhuma migration nem em init.sql — ela só existe no banco original
(mosaicfl-db) por ter sido criada manualmente antes do histórico de migrations
existir. Sem esta migration, nenhum banco novo consegue ser inicializado do zero.

Esquema reconstruído a partir do `\\d metrics.fl_checkpoints` do banco original —
apenas as colunas-base; `training_id` (migration 011) e `evaluation_json`
(migration 012) continuam sendo adicionadas depois, sem alteração.

Revision ID: 010a
Revises: 010
Create Date: 2026-07-04
"""
from typing import Sequence, Union

from alembic import op


revision: str = '010a'
down_revision: Union[str, Sequence[str], None] = '010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS metrics.fl_checkpoints (
            id          SERIAL PRIMARY KEY,
            round       INTEGER     NOT NULL,
            accuracy    REAL        NOT NULL DEFAULT 0.0,
            loss        REAL        NOT NULL DEFAULT 0.0,
            model_bytes BYTEA       NOT NULL,
            sha256      TEXT        NOT NULL,
            vocab_size  INTEGER     NOT NULL DEFAULT 0,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS metrics.fl_checkpoints;")
