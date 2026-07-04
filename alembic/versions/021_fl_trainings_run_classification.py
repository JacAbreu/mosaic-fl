"""fl_trainings — adiciona run_classification (ajuste vs. treinamento_real)

Até esta migration, a distinção entre "Treinamentos de Ajuste" (T1-T16, Bloco 1/2,
GPU, modularização, validações funcionais — training_ids 1-32) e "Treinamentos
Reais" (resultados formais para o texto de defesa, a partir de 2026-07-02) existia
só em documentação (docs/Linha_do_Tempo_MOSAIC-FL.md) — não era recuperável a
partir de um dump do banco sem essa referência externa.

Esta migration:
  1. Adiciona a coluna `run_classification` ('ajuste' | 'treinamento_real').
  2. Faz backfill explícito de todos os training_ids existentes como 'ajuste'
     (mesmo sendo o default, o backfill fica registrado e documentado aqui,
     não depende de inferência).

Novos treinamentos precisam declarar explicitamente FL_RUN_CLASSIFICATION=
treinamento_real (env var) para não caírem no default 'ajuste'.

Revision ID: 021
Revises: 020
Create Date: 2026-07-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = '021'
down_revision: Union[str, Sequence[str], None] = '020'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS run_classification TEXT NOT NULL DEFAULT 'ajuste';

        COMMENT ON COLUMN metrics.fl_trainings.run_classification IS
            '"ajuste" (tuning/debugging/validação funcional — NÃO citar como resultado '
            'final do TCC) ou "treinamento_real" (resultado formal, comparável, para o '
            'texto de defesa). Ver docs/Linha_do_Tempo_MOSAIC-FL.md, seção "Fechamento '
            'da Fase de Ajuste". Default é "ajuste" — treinamentos reais precisam '
            'declarar FL_RUN_CLASSIFICATION=treinamento_real explicitamente.';

        -- Backfill explícito: todos os training_ids conhecidos até esta migration
        -- pertencem à Fase de Ajuste (decisão da autora, 2026-07-01).
        UPDATE metrics.fl_trainings SET run_classification = 'ajuste';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS run_classification;
    """)
