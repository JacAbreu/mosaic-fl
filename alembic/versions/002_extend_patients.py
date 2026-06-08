"""extend_patients

Revision ID: 002
Revises: 001
Create Date: 2026-06-08 18:26:09.662846

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, Sequence[str], None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.patients
            ADD COLUMN IF NOT EXISTS birth_year  SMALLINT,
            ADD COLUMN IF NOT EXISTS state_code  CHAR(2),
            ADD COLUMN IF NOT EXISTS hospital_id TEXT;

        COMMENT ON COLUMN clinical.patients.birth_year  IS 'Anonymized birth year (AA_Nascimento). NULL when reported as AAAA.';
        COMMENT ON COLUMN clinical.patients.state_code  IS 'Brazilian state code (CD_UF). NULL when anonymized.';
        COMMENT ON COLUMN clinical.patients.hospital_id IS 'Source hospital: HSL | HFL | HEI | HCSP | BPSP.';
        COMMENT ON COLUMN clinical.patients.age         IS 'Age at data collection (2021 - birth_year). Kept for FL model compatibility.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.patients
            DROP COLUMN IF EXISTS birth_year,
            DROP COLUMN IF EXISTS state_code,
            DROP COLUMN IF EXISTS hospital_id;
    """)
