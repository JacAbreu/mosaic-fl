"""initial_schema

Revision ID: 001
Revises: 
Create Date: 2026-06-08 18:26:09.494911

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Applied automatically by Docker via docker-entrypoint-initdb.d/01_init.sql.
    # Reproduced here so Alembic can build the full schema on non-Docker installs.
    op.execute("""
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE SCHEMA IF NOT EXISTS clinical;
        CREATE SCHEMA IF NOT EXISTS metrics;
        CREATE SCHEMA IF NOT EXISTS knowledge;

        GRANT ALL ON SCHEMA clinical  TO mosaicfl;
        GRANT ALL ON SCHEMA metrics   TO mosaicfl;
        GRANT ALL ON SCHEMA knowledge TO mosaicfl;

        CREATE TABLE IF NOT EXISTS clinical.patients (
            patient_id  TEXT PRIMARY KEY,
            sex         TEXT NOT NULL DEFAULT 'M',
            age         REAL NOT NULL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS clinical.export_paths (
            patient_id   TEXT PRIMARY KEY,
            export_path  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clinical.fl_orchestration_config (
            id            TEXT PRIMARY KEY DEFAULT 'current',
            proximal_mu   REAL,
            pause_seconds REAL    NOT NULL DEFAULT 0.0,
            stop          BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        INSERT INTO clinical.fl_orchestration_config (id)
        VALUES ('current') ON CONFLICT DO NOTHING;

        CREATE TABLE IF NOT EXISTS metrics.risk_history (
            id          SERIAL,
            patient_id  TEXT NOT NULL,
            date        DATE NOT NULL,
            risk_score  REAL NOT NULL
        );
        SELECT create_hypertable('metrics.risk_history', 'date', if_not_exists => TRUE);

        CREATE TABLE IF NOT EXISTS metrics.exam_records (
            id           BIGSERIAL,
            patient_id   TEXT NOT NULL,
            exam_name    TEXT NOT NULL,
            date         DATE NOT NULL,
            value        REAL NOT NULL,
            phase        TEXT NOT NULL,
            ref_low      REAL NOT NULL DEFAULT 0.0,
            ref_high     REAL NOT NULL DEFAULT 0.0,
            sex_ref_low  REAL NOT NULL DEFAULT 0.0,
            sex_ref_high REAL NOT NULL DEFAULT 0.0
        );
        SELECT create_hypertable('metrics.exam_records', 'date', if_not_exists => TRUE);

        CREATE TABLE IF NOT EXISTS knowledge.clinical_profiles (
            id           TEXT PRIMARY KEY,
            document     TEXT NOT NULL,
            embedding    VECTOR(384) NOT NULL,
            desfecho     TEXT,
            faixa_etaria TEXT,
            categoria    TEXT
        );
        CREATE INDEX IF NOT EXISTS clinical_profiles_embedding_idx
            ON knowledge.clinical_profiles
            USING hnsw (embedding vector_cosine_ops);
    """)


def downgrade() -> None:
    op.execute("""
        DROP TABLE IF EXISTS knowledge.clinical_profiles;
        DROP TABLE IF EXISTS metrics.exam_records;
        DROP TABLE IF EXISTS metrics.risk_history;
        DROP TABLE IF EXISTS clinical.fl_orchestration_config;
        DROP TABLE IF EXISTS clinical.export_paths;
        DROP TABLE IF EXISTS clinical.patients;
        DROP SCHEMA IF EXISTS knowledge;
        DROP SCHEMA IF EXISTS metrics;
        DROP SCHEMA IF EXISTS clinical;
    """)
