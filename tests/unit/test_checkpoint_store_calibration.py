"""
Testes de round-trip: checkpoint salvo com calibração isotônica (calibration_method,
isotonic_calibrators, isotonic_num_classes) precisa voltar intacto do save()/load_*()
— cobre serialization.py + SQLiteCheckpointStore, a peça que torna a calibração
"parametrizável" persistível entre rodadas/processos (produção usa PostgreSQL, mas
serialization.py é compartilhado — mesmo _serialize()/_deserialize()).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.calibration import IsotonicCalibrator
from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.shared.checkpoint_store.sqlite_store import SQLiteCheckpointStore


def _fitted_calibrator(num_classes: int = 5):
    from sklearn.isotonic import IsotonicRegression
    calibrators = []
    for _ in range(num_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit([0.0, 0.5, 1.0], [0.0, 0.4, 1.0])
        calibrators.append(ir)
    return calibrators


@pytest.fixture()
def store(tmp_path):
    return SQLiteCheckpointStore(db_path=str(tmp_path / "test_calibration.db"))


class TestSaveLoadTemperature:
    def test_default_calibration_method_is_temperature(self, store):
        model = SimplifiedBEHRT()
        store.save(round_num=1, state_dict=model.state_dict(), vocab={"A": 2}, temperature=1.5)
        loaded = store.load_latest()
        assert loaded["calibration_method"] == "temperature"
        assert loaded["isotonic_calibrators"] is None

    def test_temperature_value_preserved(self, store):
        model = SimplifiedBEHRT()
        store.save(round_num=1, state_dict=model.state_dict(), vocab={"A": 2}, temperature=2.3456)
        loaded = store.load_latest()
        assert loaded["temperature"] == pytest.approx(2.3456)


class TestSaveLoadIsotonic:
    def test_isotonic_calibrators_round_trip(self, store):
        model = SimplifiedBEHRT()
        calibrators = _fitted_calibrator(num_classes=5)

        store.save(
            round_num=1,
            state_dict=model.state_dict(),
            vocab={"A": 2},
            calibration_method="isotonic",
            isotonic_calibrators=calibrators,
            isotonic_num_classes=5,
        )
        loaded = store.load_latest()

        assert loaded["calibration_method"] == "isotonic"
        assert loaded["isotonic_num_classes"] == 5
        assert len(loaded["isotonic_calibrators"]) == 5

    def test_reconstructed_calibrator_is_usable(self, store):
        """O objetivo final: reconstruir um IsotonicCalibrator funcional do checkpoint,
        sem refit — é isso que a InferenceEngine faz ao carregar."""
        model = SimplifiedBEHRT()
        calibrators = _fitted_calibrator(num_classes=5)

        store.save(
            round_num=1,
            state_dict=model.state_dict(),
            vocab={"A": 2},
            calibration_method="isotonic",
            isotonic_calibrators=calibrators,
            isotonic_num_classes=5,
        )
        loaded = store.load_latest()

        iso = IsotonicCalibrator.from_calibrators(
            loaded["isotonic_calibrators"], loaded["isotonic_num_classes"]
        )
        probs = torch.tensor([[0.3, 0.2, 0.2, 0.2, 0.1]])
        result = iso.calibrate_probs(probs)
        assert result.shape == (1, 5)
        assert abs(float(result.sum()) - 1.0) < 1e-4


class TestLoadBestPreservesCalibration:
    def test_load_best_returns_isotonic_fields(self, store):
        model = SimplifiedBEHRT()
        calibrators = _fitted_calibrator(num_classes=5)
        training_id = store.register_training()
        store.save(
            round_num=1,
            state_dict=model.state_dict(),
            vocab={"A": 2},
            accuracy=0.7,
            training_id=training_id,
            calibration_method="isotonic",
            isotonic_calibrators=calibrators,
            isotonic_num_classes=5,
        )
        loaded = store.load_best(training_id=training_id)
        assert loaded["calibration_method"] == "isotonic"
        assert len(loaded["isotonic_calibrators"]) == 5
