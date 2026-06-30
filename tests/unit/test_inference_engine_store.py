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
