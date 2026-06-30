"""evaluation_json — persiste avaliação completa do melhor checkpoint no banco

Adiciona `evaluation_json JSONB` em metrics.fl_checkpoints para armazenar a
avaliação completa do melhor checkpoint de cada treinamento: matriz de confusão,
métricas por classe, curvas de calibração e ECE isotônica.

Motivação: evaluation_round_120.json era sobrescrito a cada run, perdendo o
histórico de matrizes de confusão — dado clínico essencial para comparar
experimentos. Com este campo, cada treinamento tem sua avaliação permanente
atrelada ao checkpoint no banco.

Revision ID: 012
Revises: 011
Create Date: 2026-06-30
"""
from typing import Sequence, Union

from alembic import op


revision: str = '012'
down_revision: Union[str, Sequence[str], None] = '011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_checkpoints
            ADD COLUMN IF NOT EXISTS evaluation_json JSONB;

        COMMENT ON COLUMN metrics.fl_checkpoints.evaluation_json IS
            'Avaliação completa do checkpoint: confusion_matrix, per_class (F1/AUC/P/R), '
            'calibração pré/pós (ECE, MCE, bins), temperatura isotônica. '
            'Salvo pelo pipeline ao final do treinamento. '
            'Fonte da verdade — substitui evaluation_round_120.json (sobrescrito).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_checkpoints
            DROP COLUMN IF EXISTS evaluation_json;
    """)
