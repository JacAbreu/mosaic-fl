"""llm_judge_score/llm_judge_rationale — métrica automática COMPLEMENTAR à
avaliação Likert humana (achado 2026-07-29, decisão explícita da autora via
AskUserQuestion: automatizar a coleta e adicionar uma métrica automática,
mas NUNCA substituir a nota humana — likert_score continua null até ela
avaliar de verdade). Um LLM (mesmo backend do RAG, Ollama/gemma3:4b ou
HuggingFace fallback) julga a própria justificativa gerada, numa escala 1-5,
com o mesmo rubrica da avaliação humana. NÃO É avaliação humana — rotulado
como tal em toda citação/uso.

Revision ID: 033
Revises: 032
"""
from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.rag_likert_evaluations
            ADD COLUMN IF NOT EXISTS llm_judge_score SMALLINT,
            ADD COLUMN IF NOT EXISTS llm_judge_rationale TEXT,
            ADD COLUMN IF NOT EXISTS llm_judge_backend TEXT,
            ADD COLUMN IF NOT EXISTS llm_judge_model TEXT;
    """)
    op.execute("""
        ALTER TABLE metrics.rag_likert_evaluations
            ADD CONSTRAINT rag_llm_judge_score_range
            CHECK (llm_judge_score IS NULL OR llm_judge_score BETWEEN 1 AND 5);
    """)
    op.execute("""
        COMMENT ON COLUMN metrics.rag_likert_evaluations.llm_judge_score IS
            'Nota 1-5 automática (LLM-como-juiz) — métrica COMPLEMENTAR, nunca '
            'substitui likert_score (a nota humana de verdade). Calculada no '
            'momento da geração (--generate), sem envolvimento humano.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.rag_likert_evaluations
            DROP COLUMN IF EXISTS llm_judge_score,
            DROP COLUMN IF EXISTS llm_judge_rationale,
            DROP COLUMN IF EXISTS llm_judge_backend,
            DROP COLUMN IF EXISTS llm_judge_model;
    """)
