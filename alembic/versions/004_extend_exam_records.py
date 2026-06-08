"""extend_exam_records

Revision ID: 004
Revises: 003
Create Date: 2026-06-08 18:26:10.001897

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, Sequence[str], None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'metrics' AND table_name = 'exam_records'
                  AND column_name = 'exam_name'
            ) THEN
                ALTER TABLE metrics.exam_records RENAME COLUMN exam_name TO analyte;
            END IF;
        END $$;

        ALTER TABLE metrics.exam_records
            DROP COLUMN IF EXISTS sex_ref_low,
            DROP COLUMN IF EXISTS sex_ref_high;

        ALTER TABLE metrics.exam_records
            ADD COLUMN IF NOT EXISTS origin        TEXT,
            ADD COLUMN IF NOT EXISTS exam_group    TEXT,
            ADD COLUMN IF NOT EXISTS value_text    TEXT,
            ADD COLUMN IF NOT EXISTS unit          TEXT,
            ADD COLUMN IF NOT EXISTS attendance_id TEXT;

        COMMENT ON COLUMN metrics.exam_records.analyte        IS 'Specific measured substance (DE_Analito).';
        COMMENT ON COLUMN metrics.exam_records.exam_group     IS 'Exam family (DE_Exame): HEMOGRAMA, GASOMETRIA, etc.';
        COMMENT ON COLUMN metrics.exam_records.phase          IS 'FL model clinical phase: AB | EX | IN | OBITO | P_ALTA.';
        COMMENT ON COLUMN metrics.exam_records.origin         IS 'Sample collection context (DE_Origem): LAB | HOSP | UTI.';
        COMMENT ON COLUMN metrics.exam_records.value          IS 'Numeric result. COVID encoded: -1000=detected, -1111=negative, -1234=inconclusive.';
        COMMENT ON COLUMN metrics.exam_records.value_text     IS 'Raw result string (DE_Resultado).';
        COMMENT ON COLUMN metrics.exam_records.unit           IS 'Measurement unit (CD_Unidade).';
        COMMENT ON COLUMN metrics.exam_records.attendance_id  IS 'FK to clinical.attendances. NULL for Einstein (no ID_Atendimento).';

        CREATE INDEX IF NOT EXISTS exam_records_patient_id_idx    ON metrics.exam_records (patient_id);
        CREATE INDEX IF NOT EXISTS exam_records_attendance_id_idx ON metrics.exam_records (attendance_id);
        CREATE INDEX IF NOT EXISTS exam_records_analyte_idx       ON metrics.exam_records (analyte);
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.exam_records
            DROP COLUMN IF EXISTS origin,
            DROP COLUMN IF EXISTS exam_group,
            DROP COLUMN IF EXISTS value_text,
            DROP COLUMN IF EXISTS unit,
            DROP COLUMN IF EXISTS attendance_id;

        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'metrics' AND table_name = 'exam_records'
                  AND column_name = 'analyte'
            ) THEN
                ALTER TABLE metrics.exam_records RENAME COLUMN analyte TO exam_name;
            END IF;
        END $$;

        ALTER TABLE metrics.exam_records
            ADD COLUMN IF NOT EXISTS sex_ref_low  REAL NOT NULL DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS sex_ref_high REAL NOT NULL DEFAULT 0.0;
    """)
