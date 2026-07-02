"""fl_trainings — adiciona consumo de energia da GPU

Os logs já registram peak_ram_mb/avg_cpu_pct (migration 015) mas não consumo de
energia. Para a análise de viabilidade de implantação em ambientes com pouca
energia/água disponível para resfriamento (região do TCC), o custo energético
por treinamento precisa ser mensurável, não estimado por fora.

Coleta via `nvidia-smi --query-gpu=power.draw`, amostrada por rodada (mesma
cadência do psutil para CPU/RAM). Quando não há GPU NVIDIA disponível (CPU-only),
os campos ficam NULL — não bloqueia nem falha o treinamento.

Revision ID: 020
Revises: 019
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = '020'
down_revision: Union[str, Sequence[str], None] = '019'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS gpu_avg_power_w REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS gpu_peak_power_w REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS gpu_energy_wh REAL;

        COMMENT ON COLUMN metrics.fl_trainings.gpu_avg_power_w IS
            'Potência média da GPU em Watts, amostrada por rodada via nvidia-smi. NULL se sem GPU NVIDIA.';
        COMMENT ON COLUMN metrics.fl_trainings.gpu_peak_power_w IS
            'Pico de potência da GPU em Watts durante o treinamento. NULL se sem GPU NVIDIA.';
        COMMENT ON COLUMN metrics.fl_trainings.gpu_energy_wh IS
            'Energia total estimada em Wh (potência média × duração em horas). '
            'Estimativa por amostragem, não medição contínua — ver ressalva na documentação.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS gpu_energy_wh;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS gpu_peak_power_w;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS gpu_avg_power_w;
    """)
