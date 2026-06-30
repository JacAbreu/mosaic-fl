"""fl_round_history — histórico de métricas por rodada de cada treinamento federado

Cria metrics.fl_round_history para persistir accuracy e loss de cada rodada do loop
FL. Antes desta migration, a curva de convergência existia apenas no log de texto
(experiments/logs/*.log) e no dict history em memória — ambos descartados ou
não consultáveis de forma estruturada.

Com esta tabela é possível:
- Reconstruir a curva acc × round de qualquer training_id sem parsear logs
- Calcular o gap best_round × total_rounds diretamente via SQL
- Comparar trajetórias de convergência entre treinamentos (ex: FedAvg vs FedNova)

Revision ID: 013
Revises: 012
Create Date: 2026-06-30
"""
from typing import Sequence, Union

from alembic import op


revision: str = '013'
down_revision: Union[str, Sequence[str], None] = '012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS metrics.fl_round_history (
            id               SERIAL      PRIMARY KEY,
            training_id      INTEGER     NOT NULL
                                 REFERENCES metrics.fl_trainings(id) ON DELETE CASCADE,
            round            INTEGER     NOT NULL,
            accuracy         REAL,
            loss             REAL,
            tau_eff          REAL,
            f1_macro         REAL,
            per_class_f1     JSONB,
            round_duration_s REAL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (training_id, round)
        );

        CREATE INDEX IF NOT EXISTS fl_round_history_training_id_idx
            ON metrics.fl_round_history (training_id);

        COMMENT ON TABLE metrics.fl_round_history IS
            'Accuracy e loss de cada rodada do loop federado por training_id. '
            'Permite reconstruir a curva de convergência sem parsear logs de texto.';
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS metrics.fl_round_history_training_id_idx;
        DROP TABLE IF EXISTS metrics.fl_round_history;
    """)
