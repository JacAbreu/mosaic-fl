"""create_attendances

Revision ID: 003
Revises: 002
Create Date: 2026-06-08 18:26:09.830945

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, Sequence[str], None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS clinical.attendances (
            attendance_id    TEXT PRIMARY KEY,
            patient_id       TEXT NOT NULL REFERENCES clinical.patients(patient_id),
            hospital_id      TEXT,
            attended_at      DATE NOT NULL,
            attendance_type  TEXT,
            specialty        TEXT
        );

        COMMENT ON TABLE  clinical.attendances                 IS 'One row per hospital visit. Source: ID_Atendimento.';
        COMMENT ON COLUMN clinical.attendances.attendance_type IS 'Ambulatorial | Internado | Pronto Atendimento.';
        COMMENT ON COLUMN clinical.attendances.specialty       IS 'Clinical specialty (DE_CLINICA).';

        CREATE INDEX IF NOT EXISTS attendances_patient_id_idx  ON clinical.attendances (patient_id);
        CREATE INDEX IF NOT EXISTS attendances_attended_at_idx ON clinical.attendances (attended_at);
        CREATE INDEX IF NOT EXISTS attendances_hospital_id_idx ON clinical.attendances (hospital_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS clinical.attendances;")
