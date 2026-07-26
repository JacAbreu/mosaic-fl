"""
Testes para _build_model_metadata_note() em infrastructure/mosaicfl_api/routers/prediction.py.

Achado 2026-07-25: ModelMetadata.note tinha um texto estático fixo ("Modelo sem
calibração pós-treinamento...") independente do valor real de "calibrated" — uma
resposta real do /api/predict veio com calibrated=true E essa nota dizendo o oposto,
depois que a calibração federada (client-side fit + agregação) passou a rodar de
verdade em produção sem que o texto fosse atualizado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_api.routers.prediction import _build_model_metadata_note


class TestNotCalibrated:
    def test_mentions_no_calibration(self):
        note = _build_model_metadata_note({"calibrated": False})
        assert "sem calibração" in note

    def test_missing_calibrated_key_defaults_to_not_calibrated(self):
        note = _build_model_metadata_note({})
        assert "sem calibração" in note


class TestCalibratedTemperature:
    def test_mentions_temperature_value(self):
        note = _build_model_metadata_note({
            "calibrated": True, "calibration_method": "temperature", "temperature": 1.3149,
        })
        assert "T=1.31" in note
        assert "sem calibração" not in note

    def test_calibration_method_defaults_to_temperature(self):
        note = _build_model_metadata_note({"calibrated": True, "temperature": 2.0})
        assert "temperature scaling" in note


class TestCalibratedIsotonic:
    def test_mentions_isotonic(self):
        note = _build_model_metadata_note({"calibrated": True, "calibration_method": "isotonic"})
        assert "isotônica" in note
        assert "sem calibração" not in note


class TestConsistencyWithCalibratedFlag:
    def test_never_contradicts_calibrated_true(self):
        """A regressão real que motivou este arquivo: calibrated=True nunca pode
        coexistir com o texto que diz "sem calibração pós-treinamento"."""
        for method in ("temperature", "isotonic"):
            note = _build_model_metadata_note({
                "calibrated": True, "calibration_method": method, "temperature": 1.5,
            })
            assert "Modelo sem calibração pós-treinamento" not in note
