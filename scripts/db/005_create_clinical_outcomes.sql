-- 005_create_clinical_outcomes.sql
-- Creates metrics.clinical_outcomes to record the final result of each attendance.
--
-- outcome_class severity scale (ordinal, suitable for classification tasks):
--   0  recovered      — Alta médica curado / Alta curado
--   1  improved       — Alta melhorado / Alta médica melhorado
--   2  voluntary      — Alta a pedido (patient left without medical discharge)
--   3  transferred    — Transferência para outro hospital / serviço
--   4  ongoing        — Em atendimento / Em observação (still receiving care at cutoff)
--   5  icu            — Internado em UTI / Internação por agravamento
--   6  death          — Óbito / Óbito por COVID-19
--
-- Only HSL and BPSP provide outcome data in the FAPESP dataset.

CREATE TABLE IF NOT EXISTS metrics.clinical_outcomes (
    id               BIGSERIAL,
    patient_id       TEXT        NOT NULL,
    attendance_id    TEXT,
    outcome_at       DATE        NOT NULL,
    outcome_text     TEXT        NOT NULL,
    outcome_class    SMALLINT    NOT NULL
        CONSTRAINT outcome_class_range CHECK (outcome_class BETWEEN 0 AND 6)
);

SELECT create_hypertable(
    'metrics.clinical_outcomes', 'outcome_at',
    if_not_exists => TRUE
);

COMMENT ON TABLE  metrics.clinical_outcomes                IS 'One row per attendance outcome (desfecho clínico). Hypertable partitioned by outcome_at. Source: DESFECHOS files (HSL and BPSP only).';
COMMENT ON COLUMN metrics.clinical_outcomes.patient_id     IS 'References clinical.patients.patient_id.';
COMMENT ON COLUMN metrics.clinical_outcomes.attendance_id  IS 'References clinical.attendances.attendance_id. NULL when attendance record is absent.';
COMMENT ON COLUMN metrics.clinical_outcomes.outcome_at     IS 'Date the outcome was recorded (DT_DESFECHO).';
COMMENT ON COLUMN metrics.clinical_outcomes.outcome_text   IS 'Original outcome description (DE_DESFECHO): Alta médica curado, Óbito, etc.';
COMMENT ON COLUMN metrics.clinical_outcomes.outcome_class  IS '0=recovered | 1=improved | 2=voluntary | 3=transferred | 4=ongoing | 5=icu | 6=death';

CREATE INDEX IF NOT EXISTS clinical_outcomes_patient_id_idx    ON metrics.clinical_outcomes (patient_id);
CREATE INDEX IF NOT EXISTS clinical_outcomes_attendance_id_idx ON metrics.clinical_outcomes (attendance_id);
CREATE INDEX IF NOT EXISTS clinical_outcomes_class_idx         ON metrics.clinical_outcomes (outcome_class);
