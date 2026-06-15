"""analyte_references

Cria knowledge.analyte_references e estende metrics.exam_records com
colunas de referência canônica e classificação clínica.

Mudanças:
  knowledge.analyte_references (nova tabela)
    Armazena a média dos intervalos de referência entre hospitais
    participantes do FL, por analito e sexo. Fonte de verdade para
    computar classification em cada registro de exame.

  metrics.exam_records (extensão)
    + canonical_ref_low  — snapshot do ref canônico usado na classificação
    + canonical_ref_high — snapshot do ref canônico usado na classificação
    + classification     — resultado: HIGH | NORMAL | LOW | NO_REF

Separar analyte de classification em colunas independentes permite
treinar o modelo com hipóteses distintas sem recarregar os dados:
  - por presença do exame:      token = analyte
  - por nível clínico:          token = f"{analyte}_{classification}"
  - por padrão de anormalidade: token = classification

Os snapshots canonical_ref_low/high garantem rastreabilidade: mesmo
que analyte_references seja recalculada (novo hospital no FL), é
possível identificar quais registros usaram refs antigas e reclassificá-los.

Revision ID: 009
Revises: 008
Create Date: 2026-06-14
"""
from typing import Sequence, Union

from alembic import op


revision: str = '009'
down_revision: Union[str, Sequence[str], None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS knowledge.analyte_references (
            canonical    TEXT        NOT NULL,
            sex          CHAR(1),
            ref_low      REAL        NOT NULL,
            ref_high     REAL        NOT NULL,
            n_hospitals  SMALLINT    NOT NULL,
            source       TEXT        NOT NULL DEFAULT 'MEDIA_HOSPITAIS_PARTICIPANTES',
            computed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT analyte_references_unique UNIQUE (canonical, sex)
        );

        COMMENT ON TABLE  knowledge.analyte_references              IS 'Canonical reference ranges per analyte, derived from two-level average across FL-participating hospitals: first AVG per hospital, then AVG across hospitals. Recalculate when a new hospital joins the federation.';
        COMMENT ON COLUMN knowledge.analyte_references.canonical    IS 'Canonical analyte name — must exist as canonical in knowledge.term_dictionary (term_type=analyte).';
        COMMENT ON COLUMN knowledge.analyte_references.sex          IS 'M=male | F=female | NULL=sex-agnostic. Sex-stratified entries take precedence at ingestion when patient sex is known.';
        COMMENT ON COLUMN knowledge.analyte_references.ref_low      IS 'Canonical lower bound: AVG of per-hospital AVG(ref_low) across participating hospitals.';
        COMMENT ON COLUMN knowledge.analyte_references.ref_high     IS 'Canonical upper bound: AVG of per-hospital AVG(ref_high) across participating hospitals.';
        COMMENT ON COLUMN knowledge.analyte_references.n_hospitals  IS 'Number of hospitals used in the canonical calculation.';
        COMMENT ON COLUMN knowledge.analyte_references.source       IS 'Computation method: MEDIA_HOSPITAIS_PARTICIPANTES (default) or external source identifier.';
        COMMENT ON COLUMN knowledge.analyte_references.computed_at  IS 'Timestamp of last computation — use to detect stale classifications in exam_records.';

        CREATE INDEX IF NOT EXISTS analyte_references_canonical_idx
            ON knowledge.analyte_references (canonical, sex);

        ALTER TABLE metrics.exam_records
            ADD COLUMN IF NOT EXISTS canonical_ref_low  REAL,
            ADD COLUMN IF NOT EXISTS canonical_ref_high REAL,
            ADD COLUMN IF NOT EXISTS classification      TEXT;

        COMMENT ON COLUMN metrics.exam_records.canonical_ref_low  IS 'Snapshot of knowledge.analyte_references.ref_low at ingestion time. Preserved for traceability even if canonical refs are recalculated.';
        COMMENT ON COLUMN metrics.exam_records.canonical_ref_high IS 'Snapshot of knowledge.analyte_references.ref_high at ingestion time.';
        COMMENT ON COLUMN metrics.exam_records.classification      IS 'Clinical classification relative to canonical reference: HIGH | NORMAL | LOW | NO_REF. Computed at ingestion from value vs canonical_ref_low/high. Recompute when canonical refs change.';

        CREATE INDEX IF NOT EXISTS exam_records_classification_idx
            ON metrics.exam_records (analyte, classification)
            WHERE classification IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX  IF EXISTS metrics.exam_records_classification_idx;

        ALTER TABLE metrics.exam_records
            DROP COLUMN IF EXISTS canonical_ref_low,
            DROP COLUMN IF EXISTS canonical_ref_high,
            DROP COLUMN IF EXISTS classification;

        DROP TABLE IF EXISTS knowledge.analyte_references;
    """)
