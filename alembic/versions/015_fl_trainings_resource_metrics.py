"""fl_trainings — adiciona métricas de recurso computacional

Registra consumo de CPU, memória RAM e duração total por treinamento.
Permite comparar custo computacional entre:
  - CPU + código atual vs. CPU + código refatorado
  - CPU vs. GPU (fase futura)
  - Federado vs. centralizado (BEHRT Pooled baseline)

avg_cpu_pct pode ultrapassar 100% em sistemas multi-core (psutil Process.cpu_percent
acumula por núcleo — ex: 200% = 2 núcleos saturados).

Revision ID: 015
Revises: 014
Create Date: 2026-06-30
"""
from typing import Sequence, Union

from alembic import op


revision: str = '015'
down_revision: Union[str, Sequence[str], None] = '014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS total_duration_s REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS peak_ram_mb REAL;
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS avg_cpu_pct REAL;

        COMMENT ON COLUMN metrics.fl_trainings.total_duration_s IS
            'Duração total do loop FL em segundos (overall_start→loop_end, pré-calibração).';
        COMMENT ON COLUMN metrics.fl_trainings.peak_ram_mb IS
            'Pico de RSS do processo em MB durante o loop FL (psutil.Process.memory_info().rss).';
        COMMENT ON COLUMN metrics.fl_trainings.avg_cpu_pct IS
            'Média de CPU% do processo por rodada (psutil.Process.cpu_percent). '
            'Pode ultrapassar 100% em multi-core.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            DROP COLUMN IF EXISTS avg_cpu_pct;
        ALTER TABLE metrics.fl_trainings
            DROP COLUMN IF EXISTS peak_ram_mb;
        ALTER TABLE metrics.fl_trainings
            DROP COLUMN IF EXISTS total_duration_s;
    """)
