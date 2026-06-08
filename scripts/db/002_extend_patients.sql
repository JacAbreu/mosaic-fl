-- 002_extend_patients.sql
-- Extends clinical.patients with demographic fields available in the FAPESP dataset.
--
-- Changes:
--   + birth_year  SMALLINT  — anonymized birth year (e.g. 1963); age is kept for
--                             backwards compatibility with the FL model (computed on load)
--   + state_code  CHAR(2)   — Brazilian state abbreviation (SP, RJ, MG, ...) or NULL if anonymized
--   + hospital_id TEXT      — source institution identifier (HSL, HFL, HEI, HCSP, BPSP)

ALTER TABLE clinical.patients
    ADD COLUMN IF NOT EXISTS birth_year  SMALLINT,
    ADD COLUMN IF NOT EXISTS state_code  CHAR(2),
    ADD COLUMN IF NOT EXISTS hospital_id TEXT;

COMMENT ON COLUMN clinical.patients.birth_year  IS 'Anonymized birth year from source dataset (AA_Nascimento). NULL when reported as AAAA.';
COMMENT ON COLUMN clinical.patients.state_code  IS 'Brazilian state code (CD_UF). NULL when anonymized (MMMM).';
COMMENT ON COLUMN clinical.patients.hospital_id IS 'Source hospital identifier: HSL | HFL | HEI | HCSP | BPSP.';
COMMENT ON COLUMN clinical.patients.age         IS 'Approximate age in years at data collection (2021 - birth_year). Kept for FL model compatibility.';
