"""add_diagnosis_to_attendances

Adiciona campos de diagnóstico à tabela clinical.attendances:
  - suspected_diagnosis: hipótese diagnóstica / diagnóstico provável na admissão
  - confirmed_diagnosis:  diagnóstico definitivo confirmado (ICD-10 ou texto livre)

Esses campos são opcionais — permanecem NULL enquanto a base de dados não
fornecer essa informação (ex: base FAPESP COVID-19). Quando uma nova fonte
incluir colunas como HIPOTESE_DIAGNOSTICA ou CD_CID_DEFINITIVO, o
ColumnResolver já as mapeia automaticamente para esses campos.

Revision ID: 007
Revises: 006
Create Date: 2026-06-09
"""
from typing import Sequence, Union

from alembic import op


revision: str = '007'
down_revision: Union[str, Sequence[str], None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.attendances
            ADD COLUMN IF NOT EXISTS suspected_diagnosis TEXT,
            ADD COLUMN IF NOT EXISTS confirmed_diagnosis TEXT;

        COMMENT ON COLUMN clinical.attendances.suspected_diagnosis IS 'Working/probable diagnosis at admission (hipotese_diagnostica, CID suspeito). NULL when not provided by the data source.';
        COMMENT ON COLUMN clinical.attendances.confirmed_diagnosis IS 'Definitive confirmed diagnosis (diagnostico_confirmado, CID principal). ICD-10 code or free text. NULL when not provided by the data source.';

        CREATE INDEX IF NOT EXISTS attendances_confirmed_diagnosis_idx
            ON clinical.attendances (confirmed_diagnosis)
            WHERE confirmed_diagnosis IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS clinical.attendances_confirmed_diagnosis_idx;

        ALTER TABLE clinical.attendances
            DROP COLUMN IF EXISTS suspected_diagnosis,
            DROP COLUMN IF EXISTS confirmed_diagnosis;
    """)
