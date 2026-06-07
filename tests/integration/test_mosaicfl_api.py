"""
Testes para o módulo infrastructure/mosaicfl_api.

Cobre:
  - InferenceEngine: tokenização e predição
  - PatientDB: persistência SQLite
  - FastAPI endpoints: predict, ingest, patients, fl/status
  - Autenticação por API key
  - Watcher: detecção de novos arquivos JSON
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "integration" / "clinical-path"))

from infrastructure.mosaicfl_api.inference_engine import (  # noqa: E402
    exam_name_to_token,
    records_to_tokens,
)

# ---------------------------------------------------------------------------
# InferenceEngine — tokenização
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_token_within_vocab_range(self):
        from infrastructure.mosaicfl_api.inference_engine import VOCAB_SIZE
        tok = exam_name_to_token("WBC")
        assert 1 <= tok <= VOCAB_SIZE - 2

    def test_token_is_deterministic(self):
        assert exam_name_to_token("Hb") == exam_name_to_token("Hb")

    def test_different_exams_distinct_tokens(self):
        assert exam_name_to_token("WBC") != exam_name_to_token("Hb")

    def test_case_insensitive(self):
        assert exam_name_to_token("wbc") == exam_name_to_token("WBC")

    def test_records_to_tokens_length(self):
        class FakeRecord:
            exam_name = "WBC"
            date = date(2020, 1, 1)
        tokens = records_to_tokens([FakeRecord()] * 5, seq_len=8)
        assert len(tokens) == 8

    def test_records_to_tokens_pads_with_zeros(self):
        class FakeRecord:
            exam_name = "WBC"
            date = date(2020, 1, 1)
        tokens = records_to_tokens([FakeRecord()], seq_len=10)
        assert tokens[1:] == [0] * 9

    def test_records_to_tokens_truncates(self):
        class FakeRecord:
            exam_name = "WBC"
            date = date(2020, 1, 1)
        tokens = records_to_tokens([FakeRecord()] * 200, seq_len=16)
        assert len(tokens) == 16

    def test_empty_records_returns_all_zeros(self):
        assert records_to_tokens([], seq_len=8) == [0] * 8


# ---------------------------------------------------------------------------
# PatientDB — persistência SQLite
# ---------------------------------------------------------------------------

from infrastructure.mosaicfl_api.db import PatientDB  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    return PatientDB(tmp_path / "test.db")


class TestPatientDB:
    def test_upsert_and_list(self, db):
        db.upsert_patient("P001", "F", 55.0)
        rows = db.list_patients()
        assert any(r["patient_id"] == "P001" for r in rows)

    def test_patient_exists(self, db):
        db.upsert_patient("P002", "M", 40.0)
        assert db.patient_exists("P002")
        assert not db.patient_exists("NAOEXISTE")

    def test_add_and_get_risk(self, db):
        db.upsert_patient("P003", "F", 60.0)
        db.add_risk("P003", "2020-03-01", 0.42)
        db.add_risk("P003", "2020-03-02", 0.55)
        hist = db.get_risk_history("P003")
        assert len(hist) == 2
        assert hist[0]["risk_score"] == pytest.approx(0.42)

    def test_add_and_count_exams(self, db):
        db.upsert_patient("P004", "M", 50.0)
        db.add_exams("P004", [
            {"exam_name": "WBC", "date": "2020-03-01", "value": 8.5,
             "phase": "IN", "ref_low": 4.0, "ref_high": 11.0,
             "sex_ref_low": 4.0, "sex_ref_high": 11.0},
        ])
        assert db.exam_count("P004") == 1

    def test_set_and_get_export_path(self, db):
        db.upsert_patient("P005", "F", 35.0)
        db.set_export_path("P005", "/tmp/export/P005")
        assert db.get_export_path("P005") == "/tmp/export/P005"

    def test_export_path_none_when_not_set(self, db):
        db.upsert_patient("P006", "M", 45.0)
        assert db.get_export_path("P006") is None

    def test_latest_risk_in_list_patients(self, db):
        db.upsert_patient("P007", "F", 50.0)
        db.add_risk("P007", "2020-03-01", 0.10)
        db.add_risk("P007", "2020-03-02", 0.75)
        row = next(r for r in db.list_patients() if r["patient_id"] == "P007")
        assert row["latest_risk"] == pytest.approx(0.75)
        assert row["latest_date"] == "2020-03-02"

    def test_upsert_is_idempotent(self, db):
        db.upsert_patient("P008", "M", 30.0)
        db.upsert_patient("P008", "M", 30.0)  # segunda chamada não deve duplicar
        assert sum(1 for r in db.list_patients() if r["patient_id"] == "P008") == 1

    def test_data_persists_across_instances(self, tmp_path):
        db1 = PatientDB(tmp_path / "shared.db")
        db1.upsert_patient("P009", "F", 55.0)
        db1.add_risk("P009", "2020-03-01", 0.33)

        db2 = PatientDB(tmp_path / "shared.db")
        assert db2.patient_exists("P009")
        assert len(db2.get_risk_history("P009")) == 1


# ---------------------------------------------------------------------------
# FastAPI endpoints — TestClient com engine e DB mockados
# ---------------------------------------------------------------------------

def _make_test_client(tmp_path=None):
    import infrastructure.mosaicfl_api.service as svc

    mock_engine = MagicMock()
    mock_engine.predict.return_value = 0.42
    mock_engine.checkpoint_path = None
    svc._engine = mock_engine

    if tmp_path:
        svc._db = PatientDB(tmp_path / "test_api.db")

    from fastapi.testclient import TestClient
    return TestClient(svc.app), mock_engine, svc


@pytest.fixture(scope="module")
def client_state(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api_test")
    return _make_test_client(tmp_path=tmp)


class TestPredictEndpoint:
    def test_returns_200(self, client_state):
        client, _, _ = client_state
        r = client.post("/api/predict", json={
            "patient_id": "P001",
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.5, "phase": "IN"}]},
            headers={"X-API-Key": "secret123"},
        )
        assert r.status_code == 200

    def test_response_structure(self, client_state):
        client, _, _ = client_state
        data = client.post("/api/predict", json={
            "patient_id": "P001",
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.5, "phase": "IN"}],
        }, headers={"X-API-Key": "secret123"},).json()
        assert {"risk_score", "patient_id", "risk_date"}.issubset(data)

    def test_risk_score_is_float(self, client_state):
        client, _, _ = client_state
        data = client.post("/api/predict", json={
            "patient_id": "P001",
            "exams": [{"exam_name": "Hb", "date": "2020-03-01", "value": 12.0, "phase": "IN"}],
        }, headers={"X-API-Key": "secret123"},).json()
        assert isinstance(data["risk_score"], float)

    def test_empty_exams_returns_422(self, client_state):
        client, _, _ = client_state
        r = client.post("/api/predict", json={"patient_id": "P001", "exams": []}, headers={"X-API-Key": "secret123"},)
        assert r.status_code == 422

    def test_engine_called(self, client_state):
        client, engine, _ = client_state
        engine.predict.reset_mock()
        client.post("/api/predict", json={
            "patient_id": "P002",
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.0, "phase": "IN"}],
        }, headers={"X-API-Key": "secret123"},)
        engine.predict.assert_called_once()


class TestIngestEndpoint:
    def test_returns_200(self, client_state, tmp_path):
        client, _, _ = client_state
        r = client.post("/api/exams/ingest", json={
            "patient_id": "PI_001",
            "sex": "F",
            "age": 55.0,
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.5, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "secret123"},)
        assert r.status_code == 200

    def test_response_has_risk_and_path(self, client_state, tmp_path):
        client, _, _ = client_state
        data = client.post("/api/exams/ingest", json={
            "patient_id": "PI_002",
            "exams": [{"exam_name": "Hb", "date": "2020-03-02", "value": 13.0, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "secret123"},).json()
        assert "export_path" in data
        assert "risk_score" in data

    def test_clinicalpath_files_created(self, client_state, tmp_path):
        client, _, _ = client_state
        client.post("/api/exams/ingest", json={
            "patient_id": "PI_FILES",
            "exams": [{"exam_name": "CRP", "date": "2020-03-01", "value": 45.0, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "secret123"},)
        patient_dir = tmp_path / "Patients" / "PI_FILES"
        assert patient_dir.exists()
        for fname in ["exam-id.txt", "timestamp_to_date.txt", "node-inline-time-complete.txt"]:
            assert (patient_dir / fname).exists()

    def test_empty_exams_returns_422(self, client_state):
        client, _, _ = client_state
        r = client.post("/api/exams/ingest", json={"patient_id": "P099", "exams": []}, headers={"X-API-Key": "secret123"},)
        assert r.status_code == 422

    def test_history_persists_in_db(self, client_state, tmp_path):
        client, _, svc = client_state
        pid = "PI_PERSIST"
        for d in ["2020-03-01", "2020-03-02"]:
            client.post("/api/exams/ingest", json={
                "patient_id": pid,
                "exams": [{"exam_name": "WBC", "date": d, "value": 8.0, "phase": "IN"}],
                "output_dir": str(tmp_path),
            }, headers={"X-API-Key": "secret123"},)
        hist = svc._db.get_risk_history(pid)
        assert len(hist) == 2


class TestPatientsEndpoint:
    def test_returns_list(self, client_state):
        client, _, _ = client_state
        r = client.get("/api/patients", headers={"X-API-Key": "secret123"},)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_ingested_patient_in_list(self, client_state, tmp_path):
        client, _, _ = client_state
        pid = "PL_LIST"
        client.post("/api/exams/ingest", json={
            "patient_id": pid,
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.0, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "test-token"})
        ids = [p["patient_id"] for p in client.get("/api/patients", headers={"X-API-Key": "test-token"}).json()]
        assert pid in ids

    def test_summary_has_required_fields(self, client_state, tmp_path):
        client, _, _ = client_state
        pid = "PL_FIELDS"
        client.post("/api/exams/ingest", json={
            "patient_id": pid, "sex": "F", "age": 30.0,
            "exams": [{"exam_name": "Hb", "date": "2020-03-01", "value": 12.0, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "test-token"})
        summaries = {p["patient_id"]: p for p in client.get("/api/patients", headers={"X-API-Key": "test-token"}).json()}
        s = summaries[pid]
        assert all(k in s for k in ("latest_risk", "latest_date", "sex", "age"))


class TestPatientDetailEndpoint:
    def test_404_for_unknown(self, client_state):
        client, _, _ = client_state
        assert client.get("/api/patients/NAOEXISTE_XYZ", headers={"X-API-Key": "test-token"}).status_code == 404

    def test_risk_history_returned(self, client_state, tmp_path):
        client, _, _ = client_state
        pid = "PD_HIST"
        client.post("/api/exams/ingest", json={
            "patient_id": pid,
            "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.0, "phase": "IN"}],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "test-token"})
        data = client.get(f"/api/patients/{pid}", headers={"X-API-Key": "test-token"}).json()
        assert len(data["risk_history"]) >= 1
        assert all(k in data["risk_history"][0] for k in ("risk_score", "date"))

    def test_exam_count_correct(self, client_state, tmp_path):
        client, _, _ = client_state
        pid = "PD_COUNT"
        client.post("/api/exams/ingest", json={
            "patient_id": pid,
            "exams": [
                {"exam_name": "WBC", "date": "2020-03-01", "value": 8.0, "phase": "IN"},
                {"exam_name": "Hb",  "date": "2020-03-01", "value": 12.0, "phase": "IN"},
            ],
            "output_dir": str(tmp_path),
        }, headers={"X-API-Key": "test-token"})
        assert client.get(f"/api/patients/{pid}", headers={"X-API-Key": "test-token"}).json()["exam_count"] == 2


class TestFLStatusEndpoint:
    def test_returns_200(self, client_state):
        client, _, _ = client_state
        assert client.get("/api/fl/status").status_code == 200

    def test_required_fields(self, client_state):
        data = client_state[0].get("/api/fl/status").json()
        assert all(k in data for k in ("model_ready", "rounds_completed", "checkpoint_path"))

    def test_not_ready_without_checkpoint(self, client_state, tmp_path):
        client, _, svc = client_state
        orig = svc._CHECKPOINT_DIR
        svc._CHECKPOINT_DIR = tmp_path / "nonexistent"
        try:
            data = client.get("/api/fl/status").json()
            assert data["model_ready"] is False
            assert data["rounds_completed"] == 0
        finally:
            svc._CHECKPOINT_DIR = orig

    def test_rounds_parsed_from_filename(self, client_state, tmp_path):
        client, _, svc = client_state
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "round_7.pt").touch()
        orig = svc._CHECKPOINT_DIR
        svc._CHECKPOINT_DIR = ckpt_dir
        try:
            data = client.get("/api/fl/status").json()
            assert data["model_ready"] is True
            assert data["rounds_completed"] == 7
        finally:
            svc._CHECKPOINT_DIR = orig


class TestAuthentication:
    _EXAM = {"patient_id": "P001", "exams": [{"exam_name": "WBC", "date": "2020-03-01", "value": 8.0, "phase": "IN"}]}

    def test_any_key_accepted(self, client_state):
        """Qualquer token presente é aceito — validação de identidade é upstream."""
        client, _, _ = client_state
        r = client.post("/api/predict", json=self._EXAM, headers={"X-API-Key": "qualquer-valor"})
        assert r.status_code == 200

    def test_bearer_token_accepted(self, client_state):
        """Authorization: Bearer também é aceito como alternativa ao X-API-Key."""
        client, _, _ = client_state
        r = client.post("/api/predict", json=self._EXAM,
                        headers={"Authorization": "Bearer meu-token-hospitalar"})
        assert r.status_code == 200

    def test_missing_token_returns_403(self, client_state):
        """Sem token e FL_AUTH_REQUIRED=true (padrão) → 403."""
        client, _, _ = client_state
        r = client.post("/api/predict", json=self._EXAM)
        assert r.status_code == 403

    def test_no_auth_when_disabled(self, client_state, monkeypatch):
        """FL_AUTH_REQUIRED=false → request sem token é aceito."""
        client, _, svc = client_state
        monkeypatch.setattr(svc, "_AUTH_REQUIRED", False)
        r = client.post("/api/predict", json=self._EXAM)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class TestExamFileWatcher:
    def test_callback_on_new_json(self, tmp_path):
        _cp_dir = _ROOT / "integration" / "clinical-path"
        if str(_cp_dir) not in sys.path:
            sys.path.insert(0, str(_cp_dir))
        from watcher import ExamFileWatcher
        import threading

        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()
        received: list[Path] = []

        w = ExamFileWatcher(watch_dir, lambda p: received.append(p))
        t = threading.Thread(target=w.start, daemon=True)
        t.start()
        time.sleep(0.3)

        (watch_dir / "patient_001.json").write_text(json.dumps({"patient_id": "P001"}))
        time.sleep(0.5)
        w.stop()

        assert len(received) == 1
        assert received[0].name == "patient_001.json"

    def test_non_json_ignored(self, tmp_path):
        _cp_dir = _ROOT / "integration" / "clinical-path"
        if str(_cp_dir) not in sys.path:
            sys.path.insert(0, str(_cp_dir))
        from watcher import ExamFileWatcher
        import threading

        watch_dir = tmp_path / "incoming2"
        watch_dir.mkdir()
        received: list[Path] = []

        w = ExamFileWatcher(watch_dir, lambda p: received.append(p))
        t = threading.Thread(target=w.start, daemon=True)
        t.start()
        time.sleep(0.3)

        (watch_dir / "data.csv").write_text("exam,value")
        time.sleep(0.5)
        w.stop()

        assert received == []
