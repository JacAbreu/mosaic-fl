"""fl_trainings — adiciona partition_mode

Registra se o treinamento usou a partição non-IID natural (hospital real =
cliente) ou a partição iid_simulado (pool embaralhado, clientes virtuais) —
Experimento 3 / fase 5 do pipeline, contraste causal non-IID vs. IID.
Permite filtrar/comparar treinamentos por modo de partição via SQL direto.

Revision ID: 017
Revises: 016
Create Date: 2026-07-01
"""
from typing import Sequence, Union

from alembic import op


revision: str = '017'
down_revision: Union[str, Sequence[str], None] = '016'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS partition_mode TEXT NOT NULL DEFAULT 'natural';

        COMMENT ON COLUMN metrics.fl_trainings.partition_mode IS
            '"natural" (hospital real = cliente, non-IID) ou "iid_simulado" '
            '(pool embaralhado, clientes virtuais) — Experimento 3 / fase 5.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS partition_mode;
    """)
