"""fl_trainings — adiciona is_active_model, substitui marker file por coluna

Achado 2026-07-25/26: experiments/last_federated_training_id.txt era o único
jeito da API (infrastructure/mosaicfl_api/state.py) descobrir automaticamente
qual training_id carregar — só o Caminho A (orchestrator.py) escrevia esse
arquivo. Qualquer treino real via SuperLink/SuperNode (Caminho B, produção)
terminava e a API continuava presa ao último training_id do Caminho A, por
mais antigo/pior que fosse (achado real: API carregando checkpoint de 3
semanas atrás, accuracy=0.37, com um treino de accuracy=0.79 já pronto).

Correção imediata (2026-07-26, mesma sessão) foi replicar o mesmo mecanismo de
arquivo no Caminho B — mas arquivo local não é fonte de verdade compartilhada
entre processos/máquinas físicas diferentes (desktop+notebook). Esta migration
substitui o arquivo por uma coluna no banco, que já é compartilhado entre
server e API via FL_DB_URL.

`is_active_model=TRUE` em no máximo 1 linha por vez — garantido pela
aplicação (ProductionFedProxStrategy._mark_as_active_model(), CheckpointStore),
não por constraint de banco (não há caso de uso pra múltiplos "ativos"
simultâneos, mas não vale a pena um índice único parcial só pra isso agora).

Revision ID: 024
Revises: 023
Create Date: 2026-07-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = '024'
down_revision: Union[str, Sequence[str], None] = '023'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS is_active_model BOOLEAN NOT NULL DEFAULT FALSE;

        COMMENT ON COLUMN metrics.fl_trainings.is_active_model IS
            'TRUE para o training_id que a API de inferência deve carregar automaticamente '
            '(no máximo 1 por vez, garantido pela aplicação). Substitui '
            'experiments/last_federated_training_id.txt (arquivo local, não compartilhado '
            'entre processos/máquinas) — ver docs/Linha_do_Tempo_MOSAIC-FL.md, 2026-07-26.';

        CREATE INDEX IF NOT EXISTS fl_trainings_active_model_idx
            ON metrics.fl_trainings (is_active_model)
            WHERE is_active_model = TRUE;
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS metrics.fl_trainings_active_model_idx;
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS is_active_model;
    """)
