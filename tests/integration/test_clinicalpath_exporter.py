"""
Testes de contrato para o ClinicalPathExporter.

Verifica que os cinco arquivos gerados obedecem ao formato esperado pelo
ClinicalPath v2 e que o exame sintético FL_RISK_SCORE é injetado corretamente.
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "integration" / "clinical-path"))

from exporter import ClinicalPathExporter
from models import (
    ClinicalPhase,
    ExamRecord,
    FL_RISK_EXAM_NAME,
    FL_RISK_REF_HIGH,
    FL_RISK_REF_LOW,
    PatientExport,
    RiskPrediction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D0 = date(2020, 3, 1)
D1 = date(2020, 3, 2)
D2 = date(2020, 3, 5)


def _make_patient(with_risk: bool = True) -> PatientExport:
    records = [
        ExamRecord("WBC", D0, 8.5, ClinicalPhase.HOSPITALIZED, 4.0, 11.0, 4.0, 11.0),
        ExamRecord("Hb", D0, 12.1, ClinicalPhase.HOSPITALIZED, 12.0, 17.0, 11.0, 15.0),
        ExamRecord("WBC", D1, 9.2, ClinicalPhase.HOSPITALIZED, 4.0, 11.0, 4.0, 11.0),
        ExamRecord("CRP", D2, 45.0, ClinicalPhase.POST_DISCHARGE, 0.0, 5.0, 0.0, 5.0),
    ]
    preds = (
        [
            RiskPrediction(D0, 0.12, ClinicalPhase.HOSPITALIZED),
            RiskPrediction(D2, 0.55, ClinicalPhase.POST_DISCHARGE),
        ]
        if with_risk
        else []
    )
    return PatientExport("P001", "F", 55.0, records, preds)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("cp_export")
    patient = _make_patient(with_risk=True)
    exporter = ClinicalPathExporter()
    patient_dir = exporter.export(patient, out)
    return patient_dir, patient


# ---------------------------------------------------------------------------
# exam-id.txt
# ---------------------------------------------------------------------------


class TestExamIdFile:
    def test_file_exists(self, exported):
        patient_dir, _ = exported
        assert (patient_dir / "exam-id.txt").exists()

    def test_format_is_index_space_name(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "exam-id.txt").read_text().splitlines():
            parts = line.split(" ", 1)
            assert len(parts) == 2, f"linha mal formada: {line!r}"
            assert parts[0].isdigit(), f"índice não é inteiro: {parts[0]!r}"

    def test_indices_are_contiguous_from_zero(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        indices = [int(l.split()[0]) for l in lines]
        assert indices == list(range(len(indices)))

    def test_fl_risk_score_has_last_index(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        last_idx, last_name = lines[-1].split(" ", 1)
        assert last_name == FL_RISK_EXAM_NAME
        assert int(last_idx) == len(lines) - 1

    def test_real_exams_sorted_alphabetically(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        names = [l.split(" ", 1)[1] for l in lines if l.split(" ", 1)[1] != FL_RISK_EXAM_NAME]
        assert names == sorted(names)

    def test_no_fl_risk_score_without_predictions(self, tmp_path):
        patient = _make_patient(with_risk=False)
        exporter = ClinicalPathExporter()
        pdir = exporter.export(patient, tmp_path)
        names = [l.split(" ", 1)[1] for l in (pdir / "exam-id.txt").read_text().splitlines()]
        assert FL_RISK_EXAM_NAME not in names


# ---------------------------------------------------------------------------
# timestamp_to_date.txt
# ---------------------------------------------------------------------------


class TestTimestampToDateFile:
    def test_file_exists(self, exported):
        patient_dir, _ = exported
        assert (patient_dir / "timestamp_to_date.txt").exists()

    def test_format_is_index_space_iso_date(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "timestamp_to_date.txt").read_text().splitlines():
            parts = line.split()
            assert len(parts) == 2
            assert parts[0].isdigit()
            # ISO date parses without raising
            date.fromisoformat(parts[1])

    def test_indices_are_contiguous_from_zero(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "timestamp_to_date.txt").read_text().splitlines()
        indices = [int(l.split()[0]) for l in lines]
        assert indices == list(range(len(indices)))

    def test_dates_are_strictly_ascending(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "timestamp_to_date.txt").read_text().splitlines()
        dates = [date.fromisoformat(l.split()[1]) for l in lines]
        assert dates == sorted(dates)

    def test_all_record_dates_covered(self, exported):
        patient_dir, patient = exported
        lines = (patient_dir / "timestamp_to_date.txt").read_text().splitlines()
        mapped = {date.fromisoformat(l.split()[1]) for l in lines}
        for r in patient.exam_records:
            assert r.date in mapped
        for p in patient.risk_predictions:
            assert p.date in mapped

    def test_timestamp_zero_is_earliest_date(self, exported):
        patient_dir, patient = exported
        lines = (patient_dir / "timestamp_to_date.txt").read_text().splitlines()
        first_line = lines[0]
        ts, d = first_line.split()
        assert int(ts) == 0
        all_dates = {r.date for r in patient.exam_records} | {p.date for p in patient.risk_predictions}
        assert date.fromisoformat(d) == min(all_dates)


# ---------------------------------------------------------------------------
# time-metadata.txt
# ---------------------------------------------------------------------------


class TestTimeMetadataFile:
    def test_file_exists(self, exported):
        patient_dir, _ = exported
        assert (patient_dir / "time-metadata.txt").exists()

    def test_format_is_index_space_status(self, exported):
        patient_dir, _ = exported
        valid_statuses = {"AB", "EX", "IN", "OBITO", "P_ALTA"}
        for line in (patient_dir / "time-metadata.txt").read_text().splitlines():
            parts = line.split()
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1] in valid_statuses

    def test_no_duplicate_timestamp_status_pairs(self, exported):
        patient_dir, _ = exported
        lines = (patient_dir / "time-metadata.txt").read_text().splitlines()
        pairs = [tuple(l.split()) for l in lines]
        assert len(pairs) == len(set(pairs))

    def test_statuses_match_phases_used(self, exported):
        patient_dir, patient = exported
        lines = (patient_dir / "time-metadata.txt").read_text().splitlines()
        statuses_in_file = {l.split()[1] for l in lines}
        expected = {r.phase.status_str for r in patient.exam_records} | {
            p.phase.status_str for p in patient.risk_predictions
        }
        assert statuses_in_file == expected


# ---------------------------------------------------------------------------
# node-inline-time.txt
# ---------------------------------------------------------------------------


class TestNodeInlineTimeFile:
    def test_file_exists(self, exported):
        patient_dir, _ = exported
        assert (patient_dir / "node-inline-time.txt").exists()

    def test_format_is_three_integers(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "node-inline-time.txt").read_text().splitlines():
            parts = line.split()
            assert len(parts) == 3, f"linha mal formada: {line!r}"
            int(parts[0])
            int(parts[1])
            int(parts[2])

    def test_status_codes_are_valid(self, exported):
        patient_dir, _ = exported
        valid_codes = {-2, -1, 0, 1}
        for line in (patient_dir / "node-inline-time.txt").read_text().splitlines():
            code = int(line.split()[2])
            assert code in valid_codes

    def test_row_count_equals_records_plus_predictions(self, exported):
        patient_dir, patient = exported
        lines = (patient_dir / "node-inline-time.txt").read_text().splitlines()
        assert len(lines) == len(patient.exam_records) + len(patient.risk_predictions)

    def test_exam_ids_within_valid_range(self, exported):
        patient_dir, _ = exported
        id_lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        max_id = len(id_lines) - 1
        for line in (patient_dir / "node-inline-time.txt").read_text().splitlines():
            eid = int(line.split()[0])
            assert 0 <= eid <= max_id


# ---------------------------------------------------------------------------
# node-inline-time-complete.txt
# ---------------------------------------------------------------------------


class TestNodeInlineTimeCompleteFile:
    def test_file_exists(self, exported):
        patient_dir, _ = exported
        assert (patient_dir / "node-inline-time-complete.txt").exists()

    def test_format_is_eight_fields(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "node-inline-time-complete.txt").read_text().splitlines():
            parts = line.split()
            assert len(parts) == 8, f"linha com {len(parts)} campos: {line!r}"

    def test_row_count_matches_node_inline_time(self, exported):
        patient_dir, _ = exported
        a = (patient_dir / "node-inline-time.txt").read_text().splitlines()
        b = (patient_dir / "node-inline-time-complete.txt").read_text().splitlines()
        assert len(a) == len(b)

    def test_first_three_fields_match_node_inline_time(self, exported):
        patient_dir, _ = exported
        compact = (patient_dir / "node-inline-time.txt").read_text().splitlines()
        complete = (patient_dir / "node-inline-time-complete.txt").read_text().splitlines()
        for c_line, f_line in zip(compact, complete):
            assert f_line.startswith(c_line), (
                f"primeiros campos divergem:\n  compact:  {c_line!r}\n  complete: {f_line!r}"
            )

    def test_value_field_is_numeric(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "node-inline-time-complete.txt").read_text().splitlines():
            float(line.split()[3])

    def test_ref_range_fields_are_numeric(self, exported):
        patient_dir, _ = exported
        for line in (patient_dir / "node-inline-time-complete.txt").read_text().splitlines():
            parts = line.split()
            float(parts[4])
            float(parts[5])
            float(parts[6])
            float(parts[7])


# ---------------------------------------------------------------------------
# FL_RISK_SCORE synthetic exam
# ---------------------------------------------------------------------------


class TestFlRiskScoreExam:
    def test_risk_score_values_preserved(self, exported):
        patient_dir, patient = exported
        lines = (patient_dir / "node-inline-time-complete.txt").read_text().splitlines()
        id_lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        risk_id = next(
            int(l.split()[0]) for l in id_lines if l.split(" ", 1)[1] == FL_RISK_EXAM_NAME
        )
        risk_rows = [l for l in lines if int(l.split()[0]) == risk_id]
        assert len(risk_rows) == len(patient.risk_predictions)
        for row, pred in zip(
            sorted(risk_rows, key=lambda l: int(l.split()[1])),
            sorted(patient.risk_predictions, key=lambda p: p.date),
        ):
            assert float(row.split()[3]) == pytest.approx(pred.risk_score)

    def test_risk_score_reference_range(self, exported):
        patient_dir, _ = exported
        id_lines = (patient_dir / "exam-id.txt").read_text().splitlines()
        risk_id = next(
            int(l.split()[0]) for l in id_lines if l.split(" ", 1)[1] == FL_RISK_EXAM_NAME
        )
        complete = (patient_dir / "node-inline-time-complete.txt").read_text().splitlines()
        risk_rows = [l for l in complete if int(l.split()[0]) == risk_id]
        for row in risk_rows:
            parts = row.split()
            assert float(parts[4]) == pytest.approx(FL_RISK_REF_LOW)
            assert float(parts[5]) == pytest.approx(FL_RISK_REF_HIGH)

    def test_output_path_is_correct(self, exported):
        patient_dir, patient = exported
        assert patient_dir.name == patient.patient_id
        assert patient_dir.parent.name == "Patients"

    def test_five_files_created(self, exported):
        patient_dir, _ = exported
        expected = {
            "exam-id.txt",
            "timestamp_to_date.txt",
            "time-metadata.txt",
            "node-inline-time.txt",
            "node-inline-time-complete.txt",
        }
        created = {f.name for f in patient_dir.iterdir()}
        assert expected.issubset(created)

    def test_multiple_patients_in_separate_dirs(self, tmp_path):
        exporter = ClinicalPathExporter()
        p1 = _make_patient()
        p1.patient_id = "A001"
        p2 = _make_patient()
        p2.patient_id = "B002"
        exporter.export(p1, tmp_path)
        exporter.export(p2, tmp_path)
        assert (tmp_path / "Patients" / "A001" / "exam-id.txt").exists()
        assert (tmp_path / "Patients" / "B002" / "exam-id.txt").exists()


# ---------------------------------------------------------------------------
# ClinicalPhase contract
# ---------------------------------------------------------------------------


class TestClinicalPhaseContract:
    def test_all_phases_have_status_str(self):
        for phase in ClinicalPhase:
            assert isinstance(phase.status_str, str)
            assert len(phase.status_str) >= 2

    def test_all_phases_have_status_code(self):
        for phase in ClinicalPhase:
            assert phase.status_code in {-2, -1, 0, 1}

    def test_outpatient_maps_to_ab_minus2(self):
        assert ClinicalPhase.OUTPATIENT.status_str == "AB"
        assert ClinicalPhase.OUTPATIENT.status_code == -2

    def test_pre_hospital_maps_to_ex_minus1(self):
        assert ClinicalPhase.PRE_HOSPITAL.status_str == "EX"
        assert ClinicalPhase.PRE_HOSPITAL.status_code == -1

    def test_hospitalized_maps_to_in_zero(self):
        assert ClinicalPhase.HOSPITALIZED.status_str == "IN"
        assert ClinicalPhase.HOSPITALIZED.status_code == 0

    def test_post_discharge_maps_to_p_alta_one(self):
        assert ClinicalPhase.POST_DISCHARGE.status_str == "P_ALTA"
        assert ClinicalPhase.POST_DISCHARGE.status_code == 1
