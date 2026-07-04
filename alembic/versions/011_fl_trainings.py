"""fl_trainings

Rastreabilidade de treinamentos federados: cria metrics.fl_trainings e adiciona
training_id em metrics.fl_checkpoints com restrição UNIQUE.

Cada execução de run_training.py registra 1 linha em fl_trainings antes do loop
FL. O checkpoint guloso faz UPSERT nessa linha (1 checkpoint por treinamento).
load_best() filtra por training_id, eliminando cross-contamination entre runs
(bug observado no Exp 9: load_best() retornou R91/Exp8 em vez de R33/Exp9).

Revision ID: 011
Revises: 010a
Create Date: 2026-06-28
"""
from typing import Sequence, Union

from alembic import op


revision: str = '011'
down_revision: Union[str, Sequence[str], None] = '010a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS metrics.fl_trainings (
            id              SERIAL PRIMARY KEY,
            algorithm       TEXT        NOT NULL DEFAULT 'FedAvg',
            log_file        TEXT        NOT NULL DEFAULT '',
            n_rounds_max    INTEGER     NOT NULL DEFAULT 120,
            started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at    TIMESTAMPTZ,
            status          TEXT        NOT NULL DEFAULT 'running',
            n_rounds_done   INTEGER,
            best_round      INTEGER,
            best_accuracy   REAL,
            converged       BOOLEAN
        );

        ALTER TABLE metrics.fl_checkpoints
            ADD COLUMN IF NOT EXISTS training_id INTEGER REFERENCES metrics.fl_trainings(id);

        CREATE UNIQUE INDEX IF NOT EXISTS fl_checkpoints_training_id_uniq
            ON metrics.fl_checkpoints (training_id)
            WHERE training_id IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS metrics.fl_checkpoints_training_id_uniq;
        ALTER TABLE metrics.fl_checkpoints DROP COLUMN IF EXISTS training_id;
        DROP TABLE IF EXISTS metrics.fl_trainings;
    """)
