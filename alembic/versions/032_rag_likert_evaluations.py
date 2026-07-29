"""rag_likert_evaluations — avaliação humana (Likert 1-5) da justificativa
gerada pelo RAG, conforme previsto no plano de metodologia original (Seção
3.7, Experimento 4) mas nunca executado até 2026-07-28. Métrica de qualidade
da GERAÇÃO (a justificativa em texto é clara/correta?), diferente de
Precision@k (mede a RECUPERAÇÃO — se os casos trazidos são da classe certa).

Revision ID: 032
Revises: 031
"""
from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS metrics.rag_likert_evaluations (
            id                    SERIAL PRIMARY KEY,
            patient_id_hash       TEXT        NOT NULL,
            predicted_label       TEXT        NOT NULL,
            risk_score            REAL        NOT NULL,
            justificativa         TEXT        NOT NULL,
            fontes_json           JSONB,
            llm_backend           TEXT,
            llm_model_used        TEXT,
            llm_was_fallback      BOOLEAN     NOT NULL DEFAULT FALSE,
            alucinacao_detectada  BOOLEAN     NOT NULL DEFAULT FALSE,
            confiavel             BOOLEAN     NOT NULL DEFAULT FALSE,
            likert_score          SMALLINT,
            evaluator              TEXT        NOT NULL,
            notes                 TEXT,
            checkpoint_round      INTEGER,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT rag_likert_score_range CHECK (likert_score IS NULL OR likert_score BETWEEN 1 AND 5)
        );
    """)
    op.execute("""
        COMMENT ON TABLE metrics.rag_likert_evaluations IS
            'Avaliação humana (Likert 1-5) da justificativa gerada pelo RAG — '
            'plano original, Experimento 4, nunca executado antes de 2026-07-28. '
            'likert_score NULL = amostra gerada mas ainda não avaliada por humano.';
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS metrics.rag_likert_evaluations;")
