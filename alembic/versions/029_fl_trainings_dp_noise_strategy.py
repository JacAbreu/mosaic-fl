"""fl_trainings — adiciona dp_noise_strategy e dp_noise_group_multipliers_json

Achado 2026-07-28: DP-FedAvg (McMahan et al. 2018) foi portado do Caminho A pro
Caminho B pela primeira vez (ProductionFedProxStrategy nunca aplicava ruído DP
antes disso — só o manual_loop.py de simulação). Junto veio uma segunda estratégia
opcional de ruído, por grupo de camada (Strategy pattern, mosaicfl.core.dp_noise —
ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14/7.9).
`dp_noise_multiplier`/`dp_max_grad_norm` (migration 018) já registram o valor BASE,
mas não distinguem qual estratégia foi usada nem os multiplicadores efetivos por
grupo quando "layer_group" está ativo — essas duas colunas novas fecham essa lacuna
de auditoria.

Revision ID: 029
Revises: 028
Create Date: 2026-07-28
"""
from typing import Sequence, Union

from alembic import op


revision: str = '029'
down_revision: Union[str, Sequence[str], None] = '028'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_noise_strategy TEXT;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_noise_group_multipliers_json JSONB;

        COMMENT ON COLUMN metrics.fl_trainings.dp_noise_strategy IS
            '"uniform" (padrão, mesmo ruído em todo o modelo) ou "layer_group" '
            '(ruído diferenciado por grupo de camada — ver mosaicfl.core.dp_noise). '
            'NULL quando DP está desligado (dp_noise_multiplier=0.0).';
        COMMENT ON COLUMN metrics.fl_trainings.dp_noise_group_multipliers_json IS
            'Multiplicador de ruído efetivo por grupo de camada, na última rodada '
            'aplicada — {"embedding": 0.1, "transformer": 0.1, "head": 0.05, ...}. '
            'Sempre {"all": dp_noise_multiplier} quando dp_noise_strategy="uniform". '
            'NULL quando DP está desligado.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_noise_group_multipliers_json;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_noise_strategy;
    """)
