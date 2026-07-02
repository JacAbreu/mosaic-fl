"""fl_trainings — adiciona métricas de Differential Privacy (DP-FedAvg)

apply_dp_noise() já calculava o ε acumulado por "composição simples" (cota
gaussiana básica, McMahan et al. 2018) mas o valor era descartado — nunca
persistido. Esta migration expõe o par de contabilidades lado a lado:

  dp_epsilon_simple — cota gaussiana básica (composição sequencial simples,
                       conservadora/frouxa — já implementada)
  dp_epsilon_rdp    — Rényi Differential Privacy via opacus.RDPAccountant
                       (padrão em Opacus/TensorFlow Privacy, cota mais apertada
                       para o mesmo ruído aplicado — não muda o modelo, só a
                       precisão da prova matemática de privacidade)

Junto com dp_noise_multiplier (σ) e dp_max_grad_norm (S, clip bound) usados
naquele treino específico, para reconstruir a curva Acc×ε via SQL direto,
sem depender de re-parsing de log ou do JSONB evaluation_json.

Revision ID: 019
Revises: 018
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = '019'
down_revision: Union[str, Sequence[str], None] = '018'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_noise_multiplier REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_max_grad_norm REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_epsilon_simple REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS dp_epsilon_rdp REAL;

        COMMENT ON COLUMN metrics.fl_trainings.dp_noise_multiplier IS
            'σ usado neste treino (FL_DP_NOISE). NULL ou 0 = DP desabilitado.';
        COMMENT ON COLUMN metrics.fl_trainings.dp_max_grad_norm IS
            'S (sensitivity/clip bound do update do cliente) usado neste treino (FL_DP_CLIP).';
        COMMENT ON COLUMN metrics.fl_trainings.dp_epsilon_simple IS
            'ε acumulado via composição gaussiana simples (McMahan et al. 2018) — cota conservadora/frouxa.';
        COMMENT ON COLUMN metrics.fl_trainings.dp_epsilon_rdp IS
            'ε acumulado via Rényi DP (opacus.RDPAccountant) — cota mais apertada, mesmo mecanismo de ruído.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_epsilon_rdp;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_epsilon_simple;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_max_grad_norm;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS dp_noise_multiplier;
    """)
