"""
test_training_state_store.py — Testes de TrainingState e TrainingStateStore.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_server.state_store import TrainingState, TrainingStateStore


class TestTrainingState:

    def test_default_status_is_pending(self):
        state = TrainingState()
        assert state.status == "pending"

    def test_default_last_round_is_zero(self):
        assert TrainingState().last_round == 0

    def test_default_convergence_history_is_empty(self):
        assert TrainingState().convergence_history == []

    def test_default_timed_out_rounds_is_empty(self):
        assert TrainingState().timed_out_rounds == []


class TestTrainingStateStore:

    @pytest.fixture
    def store(self, tmp_path):
        return TrainingStateStore(tmp_path / "training_state.json")

    def test_load_returns_pending_when_file_missing(self, store):
        state = store.load()
        assert state.status == "pending"
        assert state.last_round == 0

    def test_save_creates_file(self, store, tmp_path):
        store.save(TrainingState(status="running", last_round=3))
        assert (tmp_path / "training_state.json").exists()

    def test_save_and_load_roundtrip(self, store):
        original = TrainingState(
            status="running",
            last_round=5,
            convergence_history=[0.7, 0.75, 0.78],
            converged_round=None,
        )
        store.save(original)
        recovered = store.load()
        # "running" ao carregar → "interrupted"
        assert recovered.status == "interrupted"
        assert recovered.last_round == 5
        assert recovered.convergence_history == [0.7, 0.75, 0.78]

    def test_completed_status_preserved(self, store):
        store.save(TrainingState(status="completed", last_round=10, converged_round=8))
        recovered = store.load()
        assert recovered.status == "completed"
        assert recovered.converged_round == 8

    def test_running_status_becomes_interrupted_on_load(self, store):
        store.save(TrainingState(status="running", last_round=4))
        recovered = store.load()
        assert recovered.status == "interrupted"

    def test_load_ignores_unknown_fields(self, store, tmp_path):
        path = tmp_path / "training_state.json"
        path.write_text(json.dumps({"last_round": 2, "unknown_field": "ignored"}))
        state = store.load()
        assert state.last_round == 2

    def test_load_handles_invalid_status(self, store, tmp_path):
        path = tmp_path / "training_state.json"
        path.write_text(json.dumps({"status": "bogus", "last_round": 1}))
        state = store.load()
        assert state.status == "interrupted"

    def test_load_handles_corrupted_json(self, store, tmp_path):
        path = tmp_path / "training_state.json"
        path.write_text("not valid json {{{")
        state = store.load()
        assert state.status == "pending"
        assert state.last_round == 0

    def test_save_updates_updated_at(self, store):
        state = TrainingState(status="running", last_round=1)
        old_updated_at = state.updated_at
        store.save(state)
        loaded = store.load()
        assert loaded.updated_at >= old_updated_at

    def test_timed_out_rounds_persisted(self, store):
        state = TrainingState(status="completed", last_round=10, timed_out_rounds=[3, 7])
        store.save(state)
        recovered = store.load()
        assert recovered.timed_out_rounds == [3, 7]

    def test_last_checkpoint_persisted(self, store):
        state = TrainingState(status="completed", last_checkpoint="/path/to/round_5.pt")
        store.save(state)
        recovered = store.load()
        assert recovered.last_checkpoint == "/path/to/round_5.pt"

    def test_convergence_history_fully_restored(self, store):
        history = [0.60, 0.65, 0.68, 0.70, 0.71, 0.71]
        state = TrainingState(status="completed", convergence_history=history, converged_round=5)
        store.save(state)
        recovered = store.load()
        assert recovered.convergence_history == history
        assert recovered.converged_round == 5
