"""fl_trainings — adiciona macro_auc, macro_f1, ece pós-calibração

Hoje essas métricas são calculadas ao final de cada treinamento federado (evaluate()
pós-calibração em manual_loop.py) mas só existem em texto de log e dentro do JSONB
evaluation_json de fl_checkpoints — não são consultáveis por SQL direto em fl_trainings,
ao lado de best_accuracy. Esta migration fecha essa lacuna: permite trazer o AUC-ROC
(e F1/ECE) de qualquer treinamento passado sem re-parsing de log ou de JSONB aninhado,
para justificar comparativamente a escolha de F1 macro como critério de checkpoint.

Revision ID: 016
Revises: 015
Create Date: 2026-07-01
"""
from typing import Sequence, Union

from alembic import op


revision: str = '016'
down_revision: Union[str, Sequence[str], None] = '015'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS macro_auc REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS macro_f1 REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS ece REAL;

        COMMENT ON COLUMN metrics.fl_trainings.macro_auc IS
            'AUC-ROC macro pós-calibração (temperature scaling), avaliado no melhor checkpoint restaurado.';
        COMMENT ON COLUMN metrics.fl_trainings.macro_f1 IS
            'F1 macro pós-calibração, calculado na mesma passagem que macro_auc — '
            'redundante com best_f1 (registrado só em evaluation_json), aqui fica consultável sem JSONB.';
        COMMENT ON COLUMN metrics.fl_trainings.ece IS
            'Expected Calibration Error pós-temperature-scaling. A ECE isotônica '
            '(normalmente melhor) continua só em evaluation_json — ver checkpoint_criterion.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS ece;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS macro_f1;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS macro_auc;
    """)
