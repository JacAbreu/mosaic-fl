"""rag_precision_at_k/rag_k — Precision@k do RAG portado do Caminho A pro
Caminho B (achado 2026-07-28: só existia na simulação, avaliado contra um
test_loader centralizado). No Caminho B cada cliente avalia localmente contra
sua própria val_loader real, usando uma cópia local do knowledge base
recebida do servidor (rag_patterns_json) — nunca centraliza amostra. Servidor
agrega por média ponderada (weighted_average_evaluate_metrics), mesmo padrão
de accuracy/f1_macro/macro_auc.

Revision ID: 031
Revises: 030
"""
from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS rag_precision_at_k REAL,
            ADD COLUMN IF NOT EXISTS rag_k INTEGER;
    """)
    op.execute("""
        COMMENT ON COLUMN metrics.fl_trainings.rag_precision_at_k IS
            'Fração dos k casos recuperados pelo RAG com o mesmo desfecho da '
            'consulta, agregada entre clientes (média ponderada). NULL até a '
            '2ª rodada de avaliação (precisa de um knowledge base já '
            'construído na rodada anterior) ou quando nenhum cliente enviou.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            DROP COLUMN IF EXISTS rag_precision_at_k,
            DROP COLUMN IF EXISTS rag_k;
    """)
