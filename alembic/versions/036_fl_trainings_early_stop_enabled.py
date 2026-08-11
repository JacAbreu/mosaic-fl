"""fl_trainings — adiciona early_stop_enabled

Achado 2026-08-08: FL_EARLY_STOP nunca era persistido em lugar nenhum — só
aparecia no log do SuperLink ("early_stop_enabled — servidor customizado
ativo"). Isso torna best_round/convergence_round/n_rounds_done ambíguos de
interpretar depois: sem saber se o early stop estava ligado, não dá pra saber
se "n_rounds_done < n_rounds_max" é porque o treino parou perto da
convergência (early stop) ou porque foi interrompido por outro motivo
(crash, parada manual) — casos que já se confundiram nesta mesma fase
(training_id 3/4, ver linha do tempo).

Revision ID: 036
Revises: 035
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op


revision: str = '036'
down_revision: Union[str, Sequence[str], None] = '035'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS early_stop_enabled BOOLEAN NOT NULL DEFAULT FALSE;

        COMMENT ON COLUMN metrics.fl_trainings.early_stop_enabled IS
            'Valor de FED_CFG.early_stop (env FL_EARLY_STOP) no momento em que '
            'este treino foi registrado. Quando true, n_rounds_done tende a ficar '
            'perto de convergence_round (parou por convergência, não pelo teto '
            'n_rounds_max) — sem essa coluna, não dá pra distinguir isso de uma '
            'interrupção por outro motivo (crash, parada manual) só olhando '
            'n_rounds_done < n_rounds_max.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS early_stop_enabled;
    """)
