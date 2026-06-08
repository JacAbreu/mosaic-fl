"""create_clinical_outcomes

Revision ID: 005
Revises: 004
Create Date: 2026-06-08 18:26:10.171712

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, Sequence[str], None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS metrics.clinical_outcomes (
            id               BIGSERIAL,
            patient_id       TEXT        NOT NULL,
            attendance_id    TEXT,
            outcome_at       DATE        NOT NULL,
            outcome_text     TEXT        NOT NULL,
            outcome_class    SMALLINT    NOT NULL
                CONSTRAINT outcome_class_range CHECK (outcome_class BETWEEN 0 AND 6)
        );

        SELECT create_hypertable('metrics.clinical_outcomes', 'outcome_at', if_not_exists => TRUE);

        COMMENT ON TABLE  metrics.clinical_outcomes               IS 'One row per attendance outcome. Source: DESFECHOS (HSL and BPSP only).';
        COMMENT ON COLUMN metrics.clinical_outcomes.outcome_text  IS 'Original outcome description (DE_DESFECHO).';
        COMMENT ON COLUMN metrics.clinical_outcomes.outcome_class IS '0=recovered|1=improved|2=voluntary|3=transferred|4=ongoing|5=icu|6=death';

        CREATE INDEX IF NOT EXISTS clinical_outcomes_patient_id_idx    ON metrics.clinical_outcomes (patient_id);
        CREATE INDEX IF NOT EXISTS clinical_outcomes_attendance_id_idx ON metrics.clinical_outcomes (attendance_id);
        CREATE INDEX IF NOT EXISTS clinical_outcomes_class_idx         ON metrics.clinical_outcomes (outcome_class);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS metrics.clinical_outcomes;")
