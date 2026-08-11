"""fl_trainings — CHECK constraint: is_active_model só pode ser true com status='completed'

Achado 2026-08-08: todo treino se marca is_active_model=true automaticamente ao
concluir (core.py::_save_federated_training_id_marker, "atalho de conveniência"),
sem checar validade depois — a API de inferência lê essa coluna pra decidir qual
checkpoint servir (state.py). Quando training_id=8 foi invalidado manualmente
(status='invalid', bug do evaluation_json.best_per_class_f1 achado nesta mesma
sessão), a UPDATE não limpou is_active_model — a API continuou servindo o
checkpoint de um treino já sabido problemático, sem ninguém perceber, até uma
consulta manual ao banco revelar isso.

Confirmado antes desta migration que a ordem real do código (complete_training()
sempre grava status='completed' ANTES de _save_federated_training_id_marker()
marcar is_active_model=true, em transações separadas mas sequenciais) nunca viola
esta regra no fluxo normal — o constraint só passa a rejeitar exatamente o caso
que causou o problema: alguém (código ou UPDATE manual) mudando status pra
qualquer coisa != 'completed' sem limpar is_active_model no mesmo passo.

Revision ID: 037
Revises: 036
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op


revision: str = '037'
down_revision: Union[str, Sequence[str], None] = '036'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD CONSTRAINT active_model_must_be_completed
            CHECK (is_active_model = FALSE OR status = 'completed');
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP CONSTRAINT IF EXISTS active_model_must_be_completed;
    """)
