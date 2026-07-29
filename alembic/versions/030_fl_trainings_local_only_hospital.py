"""local_only_hospital — marca treinos do Caminho B rodados com um único
hospital conectado (min-clients=1, sem federação de verdade), para servir de
baseline "local" comparável ao federado na mesma rede real de produção.

Sem isso não havia como distinguir, na rede real (Caminho B), um treino
federado de um treino local isolado — diferente do Caminho A, que filtra
hospitais via FL_INCLUDE_HOSPITALS mas também não persistia essa informação
em nenhuma tabela (só em log). Aqui persistimos desde o início.

Revision ID: 030
Revises: 029
"""
from alembic import op

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS local_only_hospital TEXT;
    """)
    op.execute("""
        COMMENT ON COLUMN metrics.fl_trainings.local_only_hospital IS
            'NULL = treino federado normal (>=2 clientes). "BPSP"/"HSL" = '
            'treino local isolado nesse hospital (min-clients=1, run-config '
            'local-only-hospital), usado como baseline pra comparação '
            'local x federado na rede real (Caminho B).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS local_only_hospital;
    """)
