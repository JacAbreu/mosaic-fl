"""fl_trainings — adiciona checkpoint_criterion

Registra o critério usado para selecionar o melhor checkpoint por rodada.
Permite comparar runs com critérios diferentes sem ambiguidade e é o pré-requisito
para a interface web de configuração (refactoring MVP) que permitirá trocar o critério
sem redeploy, com justificativa e efeitos esperados registrados.

Valores esperados: 'f1_macro' (padrão Bloco 2+), 'accuracy' (Bloco 1 — legado).

Revision ID: 014
Revises: 013
Create Date: 2026-06-30
"""
from typing import Sequence, Union

from alembic import op


revision: str = '014'
down_revision: Union[str, Sequence[str], None] = '013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS checkpoint_criterion TEXT NOT NULL DEFAULT 'f1_macro';

        COMMENT ON COLUMN metrics.fl_trainings.checkpoint_criterion IS
            'Métrica usada para selecionar o melhor checkpoint por rodada. '
            'f1_macro = Bloco 2+ (padrão); accuracy = Bloco 1 (legado). '
            'Futuro: gerenciado via fl_config com audit trail.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            DROP COLUMN IF EXISTS checkpoint_criterion;
    """)
