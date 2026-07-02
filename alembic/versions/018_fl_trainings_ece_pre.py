"""fl_trainings — adiciona ece_pre

A coluna `ece` (migration 016) registra o ECE pós-temperature-scaling. O ECE
pré-calibração (saída bruta do modelo, antes de qualquer ajuste) já era
calculado em manual_loop.py (report_raw.calibration.ece) e salvo dentro do
JSONB evaluation_json (campo pre_calibration.calibration.ece), mas sem coluna
própria — exigia navegar o JSON aninhado para comparar antes/depois. Esta
migration expõe o par pré/pós lado a lado, direto por SQL, para evidenciar o
ganho da calibração isotônica no texto de metodologia.

Revision ID: 018
Revises: 017
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = '018'
down_revision: Union[str, Sequence[str], None] = '017'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS ece_pre REAL;

        COMMENT ON COLUMN metrics.fl_trainings.ece_pre IS
            'Expected Calibration Error pré-calibração (saída bruta do modelo, '
            'antes de temperature scaling ou isotônica). Par com a coluna ece '
            '(pós-temperature-scaling) — comparação direta por SQL.';
        COMMENT ON COLUMN metrics.fl_trainings.ece IS
            'Expected Calibration Error pós-temperature-scaling (não confundir '
            'com a ECE isotônica, normalmente melhor, que continua só em '
            'evaluation_json — ver checkpoint_criterion e ece_pre).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS ece_pre;
    """)
