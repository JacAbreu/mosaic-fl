"""Testes unitários para integration/fhir — FHIRExporter e InferenceOutput."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from integration.fhir.models import InferenceOutput
from integration.fhir.mapper import FHIRExporter
from integration.fhir.loinc_map import lookup


# ── Fixtures ──────────────────────────────────────────────────────────────────

PREDICTIONS = [
    ("Alta hospitalar", 0.61),
    ("Internação prolongada", 0.22),
    ("UTI", 0.09),
    ("Óbito", 0.08),
]

@pytest.fixture
def output() -> InferenceOutput:
    return InferenceOutput(
        predictions=PREDICTIONS,
        model_round=12,
        temperature=1.24,
        ece=0.038,
        correlation_token="token-hospital-xyz",
        predicted_at=datetime(2026, 6, 24, 10, 0, 0, tzinfo=timezone.utc),
    )

@pytest.fixture
def exporter() -> FHIRExporter:
    return FHIRExporter()

@pytest.fixture
def ra(output, exporter) -> dict:
    return exporter.to_risk_assessment(output)


# ── InferenceOutput ───────────────────────────────────────────────────────────

class TestInferenceOutput:
    def test_valid_instantiation(self):
        o = InferenceOutput(predictions=PREDICTIONS, model_round=1, temperature=1.0, ece=0.04)
        assert len(o.predictions) == 4

    def test_auto_correlation_token(self):
        o = InferenceOutput(predictions=PREDICTIONS, model_round=1, temperature=1.0, ece=0.04)
        assert o.correlation_token  # não vazio
        assert len(o.correlation_token) == 36  # UUID4

    def test_auto_predicted_at(self):
        o = InferenceOutput(predictions=PREDICTIONS, model_round=1, temperature=1.0, ece=0.04)
        assert o.predicted_at.tzinfo is not None

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError, match="predictions não pode ser vazio"):
            InferenceOutput(predictions=[], model_round=1, temperature=1.0, ece=0.0)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature deve ser > 0"):
            InferenceOutput(predictions=PREDICTIONS, model_round=1, temperature=0.0, ece=0.0)

    def test_probabilities_not_summing_raises(self):
        bad = [("A", 0.5), ("B", 0.3)]  # soma = 0.8
        with pytest.raises(ValueError, match="probabilidades devem somar 1.0"):
            InferenceOutput(predictions=bad, model_round=1, temperature=1.0, ece=0.0)

    def test_probabilities_within_tolerance(self):
        # soma = 1.001 — dentro da tolerância ±0.01
        almost = [("A", 0.501), ("B", 0.500)]
        o = InferenceOutput(predictions=almost, model_round=1, temperature=1.0, ece=0.0)
        assert o.predictions


# ── FHIRExporter — estrutura FHIR R4 ─────────────────────────────────────────

class TestFHIRExporter:
    def test_resource_type(self, ra):
        assert ra["resourceType"] == "RiskAssessment"

    def test_status_final(self, ra):
        assert ra["status"] == "final"

    def test_subject_has_identifier(self, ra):
        assert "subject" in ra
        assert "identifier" in ra["subject"]

    def test_correlation_token_echoed(self, ra):
        assert ra["subject"]["identifier"]["value"] == "token-hospital-xyz"

    def test_correlation_system(self, ra):
        assert ra["subject"]["identifier"]["system"] == "urn:mosaicfl:correlation"

    def test_occurrence_date_time(self, ra):
        assert ra["occurrenceDateTime"] == "2026-06-24T10:00:00Z"

    def test_method_present(self, ra):
        assert "method" in ra
        assert ra["method"]["coding"][0]["code"] == "FedProx-BEHRT-v2"
        assert "round 12" in ra["method"]["coding"][0]["display"]

    def test_prediction_count(self, ra):
        assert len(ra["prediction"]) == 4

    def test_prediction_probabilities(self, ra):
        probs = {p["outcome"]["text"]: p["probabilityDecimal"] for p in ra["prediction"]}
        assert abs(probs["Alta hospitalar"] - 0.61) < 1e-4
        assert abs(probs["Óbito"] - 0.08) < 1e-4

    def test_prediction_outcome_coding(self, ra):
        first = ra["prediction"][0]
        assert "coding" in first["outcome"]
        assert first["outcome"]["coding"][0]["system"] == "urn:mosaicfl:outcome"
        assert first["outcome"]["coding"][0]["code"] == "alta_hospitalar"

    def test_note_contains_ece(self, ra):
        note_text = ra["note"][0]["text"]
        assert "ECE=0.0380" in note_text

    def test_note_contains_temperature(self, ra):
        note_text = ra["note"][0]["text"]
        assert "T=1.2400" in note_text

    def test_note_contains_round(self, ra):
        note_text = ra["note"][0]["text"]
        assert "round=12" in note_text

    def test_no_patient_name(self, ra):
        assert "name" not in ra
        assert "Patient" not in str(ra)

    def test_no_exam_records(self, ra):
        assert "Observation" not in str(ra)
        assert "exam_records" not in str(ra)

    def test_no_cpf_or_prontuario(self, ra):
        serialized = str(ra)
        assert "cpf" not in serialized.lower()
        assert "prontuario" not in serialized.lower()
        assert "birthDate" not in serialized

    def test_id_is_unique(self, output, exporter):
        ra1 = exporter.to_risk_assessment(output)
        ra2 = exporter.to_risk_assessment(output)
        assert ra1["id"] != ra2["id"]

    def test_profile_declared(self, ra):
        assert ra["meta"]["profile"] == [
            "http://hl7.org/fhir/StructureDefinition/RiskAssessment"
        ]


# ── Isolamento — FHIRExporter não acessa infrastructure ───────────────────────

class TestFHIRModuleIsolation:
    def test_no_infrastructure_import(self):
        import re
        import integration.fhir.mapper as mod
        import integration.fhir.models as models_mod
        import integration.fhir.loinc_map as loinc_mod

        _import_re = re.compile(r"^\s*(import|from)\s+infrastructure", re.MULTILINE)
        for m in (mod, models_mod, loinc_mod):
            source = Path(m.__file__).read_text(encoding="utf-8")
            assert not _import_re.search(source), (
                f"{m.__file__} importa infrastructure — viola isolamento do módulo FHIR"
            )

    def test_no_patient_export_import(self):
        import re
        import integration.fhir.mapper as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        _import_re = re.compile(r"^\s*(import|from)\s+\S*PatientExport|from\s+\S+\s+import\s+.*PatientExport", re.MULTILINE)
        assert not _import_re.search(source)

    def test_no_db_import(self):
        import integration.fhir.mapper as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "sqlalchemy" not in source
        assert "PatientDB" not in source


# ── LOINC map ─────────────────────────────────────────────────────────────────

class TestLOINCMap:
    def test_hemoglobina_lookup(self):
        entry = lookup("hemoglobina")
        assert entry is not None
        assert entry["code"] == "718-7"
        assert entry["system"] == "http://loinc.org"

    def test_alias_hb(self):
        assert lookup("hb") == lookup("hemoglobina")

    def test_case_insensitive(self):
        assert lookup("PCR") == lookup("pcr")

    def test_unknown_returns_none(self):
        assert lookup("analito_inexistente_xyz") is None

    def test_fl_risk_score_uses_mosaicfl_namespace(self):
        entry = lookup("fl_risk_score")
        assert entry is not None
        assert entry["system"] == "urn:mosaicfl:analyte"
