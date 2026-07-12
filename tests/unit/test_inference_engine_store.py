"""
Testes para o método InferenceEngine.load_from_store() e o fallback
state._get_engine() → CheckpointStore quando não há arquivo .pt disponível.

Estes testes cobrem o caminho de carregamento que conecta o pipeline de
treinamento (que salva no banco via CheckpointStore) à API de inferência.
"""
import io
import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "integration" / "clinical-path"))

from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.config import MODEL_CFG
from infrastructure.mosaicfl_api.inference_engine import InferenceEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_checkpoint(round_num: int = 42) -> dict:
    """Gera um dict no formato do CheckpointStore._deserialize()."""
    model = SimplifiedBEHRT()
    vocab = {"TOKEN_A": 2, "TOKEN_B": 3}
    return {
        "model_state":      model.state_dict(),
        "vocab":            vocab,
        "temperature":      1.5,
        "checkpoint_round": round_num,
        "checkpoint_at":    "2026-06-29T12:00:00+00:00",
        "model_version":    "abc123",
    }


def _fitted_isotonic_calibrators(num_classes: int = 5) -> list:
    from sklearn.isotonic import IsotonicRegression
    calibrators = []
    for _ in range(num_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit([0.0, 0.5, 1.0], [0.0, 0.4, 1.0])
        calibrators.append(ir)
    return calibrators


def _make_bare_engine() -> InferenceEngine:
    """Instância mínima de InferenceEngine, sem __init__ real — mesmo padrão usado
    em TestLoadFromStore."""
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.model = SimplifiedBEHRT()
    engine._vocab = {}
    engine._temperature = 1.0
    engine._calibration_method = "temperature"
    engine._isotonic = None
    engine._checkpoint_path = None
    engine._checkpoint_round = None
    engine._checkpoint_at = None
    engine._model_version = None
    engine._alias_cache = {}
    engine._canonical_refs = {}
    engine._mc_lock = __import__("threading").Lock()
    engine.token_mode = "FULL"
    return engine


# ---------------------------------------------------------------------------
# load_from_store()
# ---------------------------------------------------------------------------

class TestLoadFromStore:
    def test_loads_vocab(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model      = SimplifiedBEHRT()
        engine._vocab     = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        ckpt = _make_checkpoint(round_num=79)
        engine.load_from_store(ckpt)

        assert engine._vocab == {"TOKEN_A": 2, "TOKEN_B": 3}

    def test_loads_temperature(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = SimplifiedBEHRT()
        engine._vocab = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        ckpt = _make_checkpoint()
        ckpt["temperature"] = 0.75
        engine.load_from_store(ckpt)

        assert engine._temperature == pytest.approx(0.75)

    def test_loads_checkpoint_metadata(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = SimplifiedBEHRT()
        engine._vocab = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        ckpt = _make_checkpoint(round_num=79)
        engine.load_from_store(ckpt)

        assert engine._checkpoint_round == 79
        assert engine._model_version == "abc123"
        assert engine._checkpoint_path == Path("<checkpoint_store>")

    def test_raises_on_missing_vocab(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = SimplifiedBEHRT()
        engine._vocab = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        with pytest.raises(ValueError, match="vocab"):
            engine.load_from_store({"model_state": SimplifiedBEHRT().state_dict()})

    def test_raises_on_missing_model_state(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = SimplifiedBEHRT()
        engine._vocab = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        with pytest.raises(ValueError, match="model_state"):
            engine.load_from_store({"vocab": {"A": 1}})

    def test_model_weights_actually_loaded(self):
        """Garante que load_from_store() carrega os pesos, não apenas metadados."""
        model_a = SimplifiedBEHRT()
        model_b = SimplifiedBEHRT()

        # Pesos de model_a e model_b devem ser diferentes (init aleatória)
        w_a = list(model_a.parameters())[0].data.clone()
        w_b = list(model_b.parameters())[0].data.clone()
        assert not torch.allclose(w_a, w_b), "Pesos iguais por acaso — re-run o teste"

        engine = InferenceEngine.__new__(InferenceEngine)
        engine.model = model_b
        engine._vocab = {}
        engine._temperature = 1.0
        engine._checkpoint_path = None
        engine._checkpoint_round = None
        engine._checkpoint_at = None
        engine._model_version = None
        engine._alias_cache = {}
        engine._canonical_refs = {}
        engine._mc_lock = __import__("threading").Lock()
        engine.token_mode = "FULL"

        ckpt = {"model_state": model_a.state_dict(), "vocab": {"A": 2}, "temperature": 1.0}
        engine.load_from_store(ckpt)

        # Após load_from_store, os pesos de engine.model devem ser iguais aos de model_a
        w_loaded = list(engine.model.parameters())[0].data
        assert torch.allclose(w_loaded, w_a)


# ---------------------------------------------------------------------------
# state._get_engine() — fallback ao CheckpointStore
# ---------------------------------------------------------------------------

class TestGetEngineFallback:
    def test_uses_checkpoint_store_when_no_pt_file(self, tmp_path, monkeypatch):
        """Se não há round_*.pt em FL_CHECKPOINT_DIR, carrega do CheckpointStore."""
        import infrastructure.mosaicfl_api.state as state_mod

        ckpt = _make_checkpoint(round_num=55)

        mock_store = MagicMock()
        mock_store.load_best.return_value = ckpt

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            return_value=mock_store,
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        assert engine._vocab == {"TOKEN_A": 2, "TOKEN_B": 3}
        assert engine._checkpoint_round == 55

    def test_skips_store_when_pt_file_exists_and_source_is_file(self, tmp_path, monkeypatch):
        """FL_CHECKPOINT_SOURCE=file: arquivo encontrado → store não é tentado."""
        import infrastructure.mosaicfl_api.state as state_mod

        ckpt_file = tmp_path / "round_010.pt"
        model = SimplifiedBEHRT()
        buf = io.BytesIO()
        torch.save({
            "model_state":      model.state_dict(),
            "vocab":            {"FILE_TOKEN": 5},
            "temperature":      1.0,
            "checkpoint_round": 10,
            "checkpoint_at":    "2026-01-01T00:00:00+00:00",
            "model_version":    "fileversion",
        }, buf)
        ckpt_file.write_bytes(buf.getvalue())

        mock_store = MagicMock()

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_CHECKPOINT_SOURCE", "file")
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            return_value=mock_store,
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        mock_store.load_best.assert_not_called()
        assert engine._vocab == {"FILE_TOKEN": 5}

    def test_store_tried_first_when_source_is_db(self, tmp_path, monkeypatch):
        """FL_CHECKPOINT_SOURCE=db (padrão): store é tentado primeiro, mesmo com arquivo disponível."""
        import infrastructure.mosaicfl_api.state as state_mod

        ckpt_file = tmp_path / "round_010.pt"
        model = SimplifiedBEHRT()
        buf = io.BytesIO()
        torch.save({
            "model_state":      model.state_dict(),
            "vocab":            {"FILE_TOKEN": 5},
            "temperature":      1.0,
            "checkpoint_round": 10,
            "checkpoint_at":    "2026-01-01T00:00:00+00:00",
            "model_version":    "fileversion",
        }, buf)
        ckpt_file.write_bytes(buf.getvalue())

        mock_store = MagicMock()
        mock_store.load_best.return_value = None  # store vazio → cai para arquivo

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_CHECKPOINT_SOURCE", "db")
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            return_value=mock_store,
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        mock_store.load_best.assert_called_once()
        assert engine._vocab == {"FILE_TOKEN": 5}  # carregou do arquivo como fallback

    def test_load_best_called_with_training_id_when_env_set(self, tmp_path, monkeypatch):
        """FL_TRAINING_ID=5 → load_best(training_id=5): garante que a API não serve BPSP-only por engano."""
        import infrastructure.mosaicfl_api.state as state_mod

        ckpt = _make_checkpoint(round_num=79)

        mock_store = MagicMock()
        mock_store.load_best.return_value = ckpt

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_CHECKPOINT_SOURCE", "db")
        monkeypatch.setattr(state_mod, "_INFERENCE_TRAINING_ID", 5)
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            return_value=mock_store,
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        mock_store.load_best.assert_called_once_with(training_id=5)
        assert engine._checkpoint_round == 79

    def test_handles_empty_store_gracefully(self, tmp_path, monkeypatch):
        """Se o store não tem checkpoint (load_best retorna None), engine sobe sem travar."""
        import infrastructure.mosaicfl_api.state as state_mod

        mock_store = MagicMock()
        mock_store.load_best.return_value = None  # banco vazio

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            return_value=mock_store,
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        # Engine sobe sem modelo — vocab vazio, mas não lança exceção
        assert engine._vocab == {}
        assert engine._checkpoint_path is None

    def test_handles_store_error_gracefully(self, tmp_path, monkeypatch):
        """Se o banco está inacessível, engine sobe sem modelo em vez de travar."""
        import infrastructure.mosaicfl_api.state as state_mod

        monkeypatch.setattr(state_mod, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(state_mod, "_engine", None)
        monkeypatch.setenv("FL_DB_URL", "postgresql://fake/test")

        with patch(
            "infrastructure.shared.checkpoint_store.get_checkpoint_store",
            side_effect=Exception("conexão recusada"),
        ):
            with patch.object(InferenceEngine, "_load_references"):
                engine = state_mod._get_engine()

        assert engine._vocab == {}


# ---------------------------------------------------------------------------
# Calibração isotônica — carregamento via load_from_store() e aplicação em predict_proba()
# ---------------------------------------------------------------------------

class TestLoadCalibrationState:
    def test_defaults_to_temperature_when_field_absent(self):
        """Checkpoints salvos antes desta funcionalidade não têm calibration_method —
        não deve quebrar, deve manter o comportamento prévio (temperature)."""
        engine = _make_bare_engine()
        ckpt = _make_checkpoint()
        engine.load_from_store(ckpt)
        assert engine._calibration_method == "temperature"
        assert engine._isotonic is None

    def test_loads_isotonic_calibrator(self):
        engine = _make_bare_engine()
        ckpt = _make_checkpoint()
        ckpt["calibration_method"]   = "isotonic"
        ckpt["isotonic_calibrators"] = _fitted_isotonic_calibrators(num_classes=5)
        ckpt["isotonic_num_classes"] = 5

        engine.load_from_store(ckpt)

        assert engine._calibration_method == "isotonic"
        assert engine._isotonic is not None
        assert engine._isotonic._fitted is True

    def test_falls_back_to_temperature_when_isotonic_calibrators_missing(self):
        """calibration_method=isotonic mas sem os calibradores (checkpoint corrompido/parcial)
        — não deve travar, deve avisar e cair para temperature."""
        engine = _make_bare_engine()
        ckpt = _make_checkpoint()
        ckpt["calibration_method"]   = "isotonic"
        ckpt["isotonic_calibrators"] = None
        ckpt["isotonic_num_classes"] = 0

        engine.load_from_store(ckpt)

        assert engine._calibration_method == "temperature"
        assert engine._isotonic is None


class TestPredictProbaIsotonic:
    """_tokenize() é mockado — o foco aqui é a ramificação isotonic vs. temperature
    dentro de predict_proba(), não o pipeline de tokenização (já coberto em outros testes)."""

    def _tokenized_input(self, engine):
        x = torch.zeros((1, 8), dtype=torch.long)
        mask = (x == 0)
        engine._tokenize = MagicMock(return_value=(x, mask))
        return [{"exam_name": "TOKEN_A", "date": "2026-01-01", "value": 1.0}]

    def test_uses_isotonic_when_active(self):
        """Com calibrador isotônico ativo, predict_proba deve chamar calibrate_probs()
        em vez de dividir logits por temperatura, e a saída deve seguir sendo uma
        distribuição de probabilidade válida por classe."""
        engine = _make_bare_engine()
        engine._vocab = {"TOKEN_A": 2}
        engine._calibration_method = "isotonic"
        engine._isotonic = MagicMock()
        engine._isotonic.calibrate_probs.side_effect = lambda probs: probs  # passthrough

        records = self._tokenized_input(engine)
        result = engine.predict_proba(records, mc_samples=3)

        assert engine._isotonic.calibrate_probs.called
        assert result["mc_samples"] == 3
        total = sum(v["value"] for v in result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_calibrated_flag_true_when_isotonic_fitted(self):
        engine = _make_bare_engine()
        engine._vocab = {"TOKEN_A": 2}
        engine._calibration_method = "isotonic"
        engine._isotonic = MagicMock()
        engine._isotonic.calibrate_probs.side_effect = lambda probs: probs

        records = self._tokenized_input(engine)
        result = engine.predict_proba(records, mc_samples=2)

        assert result["calibrated"] is True
        assert result["calibration_method"] == "isotonic"

    def test_temperature_path_unaffected_by_isotonic_fields(self):
        """Regressão: garantir que o comportamento prévio (temperature) continua intacto
        quando calibration_method="temperature" (isotonic nunca é chamado)."""
        engine = _make_bare_engine()
        engine._vocab = {"TOKEN_A": 2}
        engine._temperature = 1.3
        engine._calibration_method = "temperature"
        engine._isotonic = MagicMock()  # não deve ser chamado

        records = self._tokenized_input(engine)
        result = engine.predict_proba(records, mc_samples=2)

        engine._isotonic.calibrate_probs.assert_not_called()
        assert result["calibration_method"] == "temperature"
        total = sum(v["value"] for v in result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.05)
