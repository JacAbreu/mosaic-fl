-- 004_extend_exam_records.sql
-- Extends metrics.exam_records with richer fields from the FAPESP dataset.
--
-- Changes:
--   exam_name  → renamed to analyte  (specific measured substance, e.g. Leucócitos)
--   phase        kept as-is          (FL model clinical phase: AB | EX | IN | OBITO | P_ALTA)
--   + origin         TEXT NULL — sample collection context from FAPESP (DE_Origem: LAB | HOSP | UTI)
--   + exam_group     TEXT      — exam family (DE_Exame, e.g. HEMOGRAMA, GASOMETRIA)
--   + value_text     TEXT      — raw result as received (DE_Resultado, before numeric extraction)
--   + unit           TEXT      — measurement unit (CD_Unidade, e.g. g/dL, mg/dL, %)
--   + attendance_id  TEXT NULL — FK to clinical.attendances (NULL for Einstein: no ID_Atendimento)
--   - sex_ref_low    removed   (not present in source data)
--   - sex_ref_high   removed

-- Rename exam_name → analyte
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'metrics'
          AND table_name   = 'exam_records'
          AND column_name  = 'exam_name'
    ) THEN
        ALTER TABLE metrics.exam_records RENAME COLUMN exam_name TO analyte;
    END IF;
END
$$;

-- Drop unused columns
ALTER TABLE metrics.exam_records
    DROP COLUMN IF EXISTS sex_ref_low,
    DROP COLUMN IF EXISTS sex_ref_high;

-- Add new columns
ALTER TABLE metrics.exam_records
    ADD COLUMN IF NOT EXISTS origin        TEXT,
    ADD COLUMN IF NOT EXISTS exam_group    TEXT,
    ADD COLUMN IF NOT EXISTS value_text    TEXT,
    ADD COLUMN IF NOT EXISTS unit          TEXT,
    ADD COLUMN IF NOT EXISTS attendance_id TEXT;

COMMENT ON TABLE  metrics.exam_records                IS 'One row per lab or clinical measurement. Hypertable partitioned by date.';
COMMENT ON COLUMN metrics.exam_records.analyte        IS 'Specific measured substance (DE_Analito): Leucócitos, Hemoglobina, Glicose, etc.';
COMMENT ON COLUMN metrics.exam_records.exam_group     IS 'Exam family (DE_Exame): HEMOGRAMA, GASOMETRIA, PCR, SARS-CoV-2, etc.';
COMMENT ON COLUMN metrics.exam_records.date           IS 'Sample collection date (DT_Coleta).';
COMMENT ON COLUMN metrics.exam_records.phase          IS 'FL model clinical phase: AB=ambulatório | EX=externo | IN=internado | OBITO | P_ALTA.';
COMMENT ON COLUMN metrics.exam_records.origin         IS 'Sample collection context (DE_Origem): LAB | HOSP | UTI. NULL when not provided.';
COMMENT ON COLUMN metrics.exam_records.value          IS 'Numeric result. Qualitative COVID results encoded: -1000=detected, -1111=negative, -1234=inconclusive, -2222=other.';
COMMENT ON COLUMN metrics.exam_records.value_text     IS 'Raw result string as received (DE_Resultado), before numeric extraction.';
COMMENT ON COLUMN metrics.exam_records.unit           IS 'Measurement unit (CD_Unidade): g/dL, mg/dL, %, etc.';
COMMENT ON COLUMN metrics.exam_records.ref_low        IS 'Lower bound of reference range (parsed from DE_VALOR_REFERENCIA). 0.0 when not available.';
COMMENT ON COLUMN metrics.exam_records.ref_high       IS 'Upper bound of reference range (parsed from DE_VALOR_REFERENCIA). 0.0 when not available.';
COMMENT ON COLUMN metrics.exam_records.attendance_id  IS 'FK to clinical.attendances. NULL for Einstein dataset (no ID_Atendimento in exams).';

CREATE INDEX IF NOT EXISTS exam_records_patient_id_idx    ON metrics.exam_records (patient_id);
CREATE INDEX IF NOT EXISTS exam_records_attendance_id_idx ON metrics.exam_records (attendance_id);
CREATE INDEX IF NOT EXISTS exam_records_analyte_idx       ON metrics.exam_records (analyte);
