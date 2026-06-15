-- 009_analyte_references.sql
-- Cria knowledge.analyte_references e estende metrics.exam_records.
--
-- Ver alembic/versions/009_analyte_references.py para documentação completa.

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

COMMENT ON TABLE  knowledge.analyte_references              IS 'Canonical reference ranges per analyte, derived from two-level average across FL-participating hospitals.';
COMMENT ON COLUMN knowledge.analyte_references.canonical    IS 'Canonical analyte name — must exist in knowledge.term_dictionary (term_type=analyte).';
COMMENT ON COLUMN knowledge.analyte_references.sex          IS 'M | F | NULL (sex-agnostic). Sex-stratified entries take precedence when patient sex is known.';
COMMENT ON COLUMN knowledge.analyte_references.ref_low      IS 'AVG of per-hospital AVG(ref_low) across participating hospitals.';
COMMENT ON COLUMN knowledge.analyte_references.ref_high     IS 'AVG of per-hospital AVG(ref_high) across participating hospitals.';
COMMENT ON COLUMN knowledge.analyte_references.n_hospitals  IS 'Number of hospitals used in the calculation.';
COMMENT ON COLUMN knowledge.analyte_references.computed_at  IS 'Last computation timestamp — use to detect stale classifications in exam_records.';

CREATE INDEX IF NOT EXISTS analyte_references_canonical_idx
    ON knowledge.analyte_references (canonical, sex);

ALTER TABLE metrics.exam_records
    ADD COLUMN IF NOT EXISTS canonical_ref_low  REAL,
    ADD COLUMN IF NOT EXISTS canonical_ref_high REAL,
    ADD COLUMN IF NOT EXISTS classification      TEXT;

COMMENT ON COLUMN metrics.exam_records.canonical_ref_low  IS 'Snapshot of analyte_references.ref_low at ingestion time.';
COMMENT ON COLUMN metrics.exam_records.canonical_ref_high IS 'Snapshot of analyte_references.ref_high at ingestion time.';
COMMENT ON COLUMN metrics.exam_records.classification      IS 'HIGH | NORMAL | LOW | NO_REF — computed at ingestion from value vs canonical_ref_low/high.';

CREATE INDEX IF NOT EXISTS exam_records_classification_idx
    ON metrics.exam_records (analyte, classification)
    WHERE classification IS NOT NULL;
