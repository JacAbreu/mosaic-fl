"""extend_patients_attendances

Formaliza colunas adicionadas fora do ciclo de migrations (patch_new_fields):
  - clinical.patients:    municipality, cep_prefix
  - clinical.attendances: clinic_id

Revision ID: 006
Revises: 005
Create Date: 2026-06-09
"""
from typing import Sequence, Union

from alembic import op


revision: str = '006'
down_revision: Union[str, Sequence[str], None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.patients
            ADD COLUMN IF NOT EXISTS municipality TEXT,
            ADD COLUMN IF NOT EXISTS cep_prefix   VARCHAR(5);

        COMMENT ON COLUMN clinical.patients.municipality IS 'City name (CD_Municipio). Populated for HSL, BPSP, HFL, HCSP; NULL for HEI due to overflow truncation.';
        COMMENT ON COLUMN clinical.patients.cep_prefix   IS 'First 5 digits of postal code (CD_CepReduzido). Truncated to VARCHAR(5).';

        ALTER TABLE clinical.attendances
            ADD COLUMN IF NOT EXISTS clinic_id TEXT;

        COMMENT ON COLUMN clinical.attendances.clinic_id IS 'Internal clinic/ward identifier (ID_Clinica). Available for HSL and BPSP only.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.patients
            DROP COLUMN IF EXISTS municipality,
            DROP COLUMN IF EXISTS cep_prefix;

        ALTER TABLE clinical.attendances
            DROP COLUMN IF EXISTS clinic_id;
    """)
