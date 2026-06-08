-- 003_create_attendances.sql
-- Creates clinical.attendances to represent each hospital visit (encounter).
--
-- An attendance links a patient to a specific visit at a specific hospital.
-- Exam results and clinical outcomes reference this table.
-- Note: Einstein dataset does not provide attendance IDs for exams — those rows
--       will have exam_records.attendance_id = NULL.

CREATE TABLE IF NOT EXISTS clinical.attendances (
    attendance_id    TEXT        PRIMARY KEY,
    patient_id       TEXT        NOT NULL REFERENCES clinical.patients(patient_id),
    hospital_id      TEXT,
    attended_at      DATE        NOT NULL,
    attendance_type  TEXT,
    specialty        TEXT
);

COMMENT ON TABLE  clinical.attendances                  IS 'One row per hospital visit (atendimento). Source: ID_Atendimento.';
COMMENT ON COLUMN clinical.attendances.attendance_id    IS 'Anonymized visit ID (ID_Atendimento, 32-char hash).';
COMMENT ON COLUMN clinical.attendances.patient_id       IS 'References clinical.patients.patient_id.';
COMMENT ON COLUMN clinical.attendances.hospital_id      IS 'Hospital that performed this visit: HSL | HFL | HEI | HCSP | BPSP.';
COMMENT ON COLUMN clinical.attendances.attended_at      IS 'Visit date (DT_Atendimento).';
COMMENT ON COLUMN clinical.attendances.attendance_type  IS 'Visit modality: Ambulatorial | Internado | Pronto Atendimento.';
COMMENT ON COLUMN clinical.attendances.specialty        IS 'Clinical specialty (DE_CLINICA): Cardiologia, UTI, Clínica Médica, etc.';

CREATE INDEX IF NOT EXISTS attendances_patient_id_idx  ON clinical.attendances (patient_id);
CREATE INDEX IF NOT EXISTS attendances_attended_at_idx ON clinical.attendances (attended_at);
CREATE INDEX IF NOT EXISTS attendances_hospital_id_idx ON clinical.attendances (hospital_id);
