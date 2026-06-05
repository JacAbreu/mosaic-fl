"""
test_config_loader.py
Testes unitários para infrastructure/mosaicfl_server/config_loader.py.

Cobre:
  - ChromaDBConfigLoader._cast   : coerção de tipos do metadata ChromaDB
  - ChromaDBConfigLoader.load    : leitura e tratamento de erros
  - ChromaDBConfigLoader.write   : persistência e filtragem de tipos
  - ChromaDBConfigLoader.clear   : remoção do documento
  - FileConfigLoader.load        : leitura, arquivo ausente, JSON inválido
  - FileConfigLoader.write       : criação de diretórios, conteúdo correto
  - FileConfigLoader.clear       : remoção de arquivo
  - get_config_loader            : seleção de backend via FL_CONFIG_BACKEND

Uso:
    pytest tests/test_config_loader.py -v
    pytest tests/test_config_loader.py -v -k "TestCast"
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.mosaicfl_server.config_loader import (
    ChromaDBConfigLoader,
    FileConfigLoader,
    get_config_loader,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDBConfigLoader._cast
# ═══════════════════════════════════════════════════════════════════════════════

class TestCast:
    """
    _cast converte os valores de metadata do ChromaDB (sempre strings ou primitivos)
    para os tipos corretos em Python. É a lógica mais crítica do loader — erros aqui
    causam stop=False quando deveria ser True, ou proximal_mu silenciosamente ignorado.
    """

    def test_stop_true_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "true"})["stop"] is True

    def test_stop_True_capitalized(self):
        assert ChromaDBConfigLoader._cast({"stop": "True"})["stop"] is True

    def test_stop_one_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "1"})["stop"] is True

    def test_stop_yes_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "yes"})["stop"] is True

    def test_stop_false_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "false"})["stop"] is False

    def test_stop_zero_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "0"})["stop"] is False

    def test_stop_native_bool_true(self):
        assert ChromaDBConfigLoader._cast({"stop": True})["stop"] is True

    def test_stop_native_bool_false(self):
        assert ChromaDBConfigLoader._cast({"stop": False})["stop"] is False

    def test_proximal_mu_string_float(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": "0.005"})
        assert result["proximal_mu"] == pytest.approx(0.005)

    def test_proximal_mu_native_float(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": 0.01})
        assert result["proximal_mu"] == pytest.approx(0.01)

    def test_proximal_mu_invalid_string_omitted(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": "nao_e_numero"})
        assert "proximal_mu" not in result

    def test_proximal_mu_none_omitted(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": None})
        assert "proximal_mu" not in result

    def test_pause_seconds_string_float(self):
        result = ChromaDBConfigLoader._cast({"pause_seconds": "30.5"})
        assert result["pause_seconds"] == pytest.approx(30.5)

    def test_pause_seconds_invalid_string_omitted(self):
        result = ChromaDBConfigLoader._cast({"pause_seconds": "abc"})
        assert "pause_seconds" not in result

    def test_unknown_key_passes_through(self):
        result = ChromaDBConfigLoader._cast({"custom_key": "valor"})
        assert result["custom_key"] == "valor"

    def test_empty_dict_returns_empty(self):
        assert ChromaDBConfigLoader._cast({}) == {}

    def test_mixed_keys(self):
        raw = {"stop": "true", "proximal_mu": "0.01", "pause_seconds": "5.0", "note": "test"}
        result = ChromaDBConfigLoader._cast(raw)
        assert result["stop"] is True
        assert result["proximal_mu"] == pytest.approx(0.01)
        assert result["pause_seconds"] == pytest.approx(5.0)
        assert result["note"] == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDBConfigLoader.load
# ═══════════════════════════════════════════════════════════════════════════════

class TestChromaDBLoad:

    def _make_loader(self, metadata=None, raise_on_get=False):
        """Cria ChromaDBConfigLoader com ChromaDB completamente mockado."""
        mock_collection = MagicMock()
        if raise_on_get:
            mock_collection.get.side_effect = Exception("ChromaDB indisponível")
        elif metadata is None:
            mock_collection.get.return_value = {"metadatas": [None]}
        else:
            mock_collection.get.return_value = {"metadatas": [metadata]}

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client):
            loader = ChromaDBConfigLoader(db_path="/fake/path")
        return loader, mock_collection

    def test_empty_collection_returns_empty_dict(self):
        loader, _ = self._make_loader(metadata=None)
        assert loader.load(round_num=1) == {}

    def test_no_metadatas_key_returns_empty_dict(self):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"metadatas": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("chromadb.PersistentClient", return_value=mock_client):
            loader = ChromaDBConfigLoader(db_path="/fake/path")
        assert loader.load(round_num=1) == {}

    def test_load_returns_parsed_config(self):
        loader, _ = self._make_loader({"proximal_mu": "0.01", "stop": "false"})
        result = loader.load(round_num=2)
        assert result["proximal_mu"] == pytest.approx(0.01)
        assert result["stop"] is False

    def test_load_stop_true_is_bool(self):
        loader, _ = self._make_loader({"stop": "true"})
        result = loader.load(round_num=1)
        assert result["stop"] is True
        assert isinstance(result["stop"], bool)

    def test_load_exception_returns_empty_dict(self):
        loader, _ = self._make_loader(raise_on_get=True)
        result = loader.load(round_num=5)
        assert result == {}

    def test_load_calls_get_with_correct_id(self):
        loader, mock_collection = self._make_loader(metadata={})
        loader.load(round_num=3)
        mock_collection.get.assert_called_once_with(ids=["runtime_config"])

    def test_load_round_num_does_not_affect_result(self):
        loader, _ = self._make_loader({"proximal_mu": "0.005"})
        r1 = loader.load(round_num=1)
        r2 = loader.load(round_num=99)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDBConfigLoader.write
# ═══════════════════════════════════════════════════════════════════════════════

class TestChromaDBWrite:

    def _make_loader(self):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("chromadb.PersistentClient", return_value=mock_client):
            loader = ChromaDBConfigLoader(db_path="/fake/path")
        return loader, mock_collection

    def test_write_calls_upsert(self):
        loader, mock_collection = self._make_loader()
        loader.write({"proximal_mu": 0.01, "stop": False})
        mock_collection.upsert.assert_called_once()

    def test_write_uses_fixed_doc_id(self):
        loader, mock_collection = self._make_loader()
        loader.write({"stop": True})
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["runtime_config"]

    def test_write_filters_non_primitive_types(self):
        loader, mock_collection = self._make_loader()
        loader.write({"proximal_mu": 0.01, "nested": {"a": 1}, "lista": [1, 2]})
        call_kwargs = mock_collection.upsert.call_args[1]
        metadata = call_kwargs["metadatas"][0]
        assert "nested" not in metadata
        assert "lista" not in metadata
        assert "proximal_mu" in metadata

    def test_write_accepts_bool_values(self):
        loader, mock_collection = self._make_loader()
        loader.write({"stop": True})
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["stop"] is True

    def test_write_exception_does_not_propagate(self):
        loader, mock_collection = self._make_loader()
        mock_collection.upsert.side_effect = Exception("falha de escrita")
        loader.write({"stop": True})  # não deve lançar


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDBConfigLoader.clear
# ═══════════════════════════════════════════════════════════════════════════════

class TestChromaDBClear:

    def _make_loader(self):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("chromadb.PersistentClient", return_value=mock_client):
            loader = ChromaDBConfigLoader(db_path="/fake/path")
        return loader, mock_collection

    def test_clear_calls_delete_with_correct_id(self):
        loader, mock_collection = self._make_loader()
        loader.clear()
        mock_collection.delete.assert_called_once_with(ids=["runtime_config"])

    def test_clear_exception_does_not_propagate(self):
        loader, mock_collection = self._make_loader()
        mock_collection.delete.side_effect = Exception("falha")
        loader.clear()  # não deve lançar


# ═══════════════════════════════════════════════════════════════════════════════
# FileConfigLoader.load
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileConfigLoaderLoad:

    def test_missing_file_returns_empty_dict(self, tmp_path):
        loader = FileConfigLoader(path=tmp_path / "nao_existe.json")
        assert loader.load(round_num=1) == {}

    def test_valid_json_returns_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"proximal_mu": 0.005, "stop": False}))
        loader = FileConfigLoader(path=config_file)
        result = loader.load(round_num=1)
        assert result["proximal_mu"] == pytest.approx(0.005)
        assert result["stop"] is False

    def test_corrupted_json_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text("{ chave: sem aspas }")
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == {}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text("")
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == {}

    def test_load_roundtrip_preserves_types(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        original = {"proximal_mu": 0.01, "pause_seconds": 30.0, "stop": True}
        config_file.write_text(json.dumps(original))
        loader = FileConfigLoader(path=config_file)
        result = loader.load(round_num=2)
        assert result == original

    def test_load_result_independent_of_round_num(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"stop": False}))
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == loader.load(round_num=99)


# ═══════════════════════════════════════════════════════════════════════════════
# FileConfigLoader.write
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileConfigLoaderWrite:

    def test_write_creates_file(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": False})
        assert config_file.exists()

    def test_write_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "runtime_config.json"
        loader = FileConfigLoader(path=nested)
        loader.write({"proximal_mu": 0.01})
        assert nested.exists()

    def test_write_content_is_valid_json(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"proximal_mu": 0.005, "stop": True})
        with open(config_file) as f:
            data = json.load(f)
        assert data["proximal_mu"] == pytest.approx(0.005)
        assert data["stop"] is True

    def test_write_then_load_roundtrip(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        original = {"proximal_mu": 0.01, "pause_seconds": 5.0, "stop": False}
        loader.write(original)
        assert loader.load(round_num=1) == original

    def test_write_overwrites_previous_config(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": False, "proximal_mu": 0.01})
        loader.write({"stop": True})
        result = loader.load(round_num=1)
        assert result["stop"] is True
        assert "proximal_mu" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# FileConfigLoader.clear
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileConfigLoaderClear:

    def test_clear_removes_file(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"stop": True}))
        loader = FileConfigLoader(path=config_file)
        loader.clear()
        assert not config_file.exists()

    def test_clear_nonexistent_file_does_not_raise(self, tmp_path):
        loader = FileConfigLoader(path=tmp_path / "nao_existe.json")
        loader.clear()  # não deve lançar

    def test_clear_then_load_returns_empty(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": True})
        loader.clear()
        assert loader.load(round_num=1) == {}


# ═══════════════════════════════════════════════════════════════════════════════
# get_config_loader — seleção de backend
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetConfigLoader:

    def test_file_backend_returns_file_loader(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "file")
        loader = get_config_loader()
        assert isinstance(loader, FileConfigLoader)

    def test_file_backend_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "FILE")
        loader = get_config_loader()
        assert isinstance(loader, FileConfigLoader)

    def test_chroma_backend_returns_chroma_loader(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "chroma")
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)

    def test_default_backend_is_chroma(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FL_CONFIG_BACKEND", raising=False)
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)

    def test_unknown_backend_falls_back_to_chroma(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FL_CONFIG_BACKEND", "redis")
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path))
        loader = get_config_loader()
        assert isinstance(loader, ChromaDBConfigLoader)


# ═══════════════════════════════════════════════════════════════════════════════
# Integração: FileConfigLoader como ConfigLoader na ProductionFedProxStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigLoaderIntegrationWithStrategy:
    """
    Verifica que configure_fit da strategy aplica corretamente o config
    retornado por um FileConfigLoader real (sem mock de ChromaDB).
    """

    @pytest.fixture
    def strategy_with_file_loader(self, tmp_path):
        from infrastructure.mosaicfl_server.strategy import ProductionFedProxStrategy
        from mosaicfl.v2.model_v2 import SimplifiedBEHRT

        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        model = SimplifiedBEHRT(use_cls_token=True)

        with patch("flwr.server.strategy.FedProx.__init__", return_value=None):
            strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
            strategy.global_model = model
            strategy.config_loader = loader
            strategy.on_round_start = None
            strategy.proximal_mu = 0.01
            strategy.should_stop = False
            from infrastructure.mosaicfl_server.strategy import ConvergenceTracker
            strategy.tracker = ConvergenceTracker()
            strategy.round_counter = 0
            (tmp_path / "checkpoints").mkdir()
            (tmp_path / "logs").mkdir()

        import infrastructure.mosaicfl_server.strategy as strat_mod
        strat_mod.CHECKPOINT_DIR = tmp_path / "checkpoints"
        strat_mod.LOG_DIR = tmp_path / "logs"

        return strategy, loader

    def test_configure_fit_returns_empty_when_stop_true(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        loader.write({"stop": True})
        result = strategy.configure_fit(1, MagicMock(), MagicMock())
        assert result == []
        assert strategy.should_stop is True

    def test_configure_fit_updates_proximal_mu(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        loader.write({"proximal_mu": 0.05, "stop": False})
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(1, MagicMock(), MagicMock())
        assert strategy.proximal_mu == pytest.approx(0.05)

    def test_configure_fit_no_config_delegates_to_super(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        # sem arquivo de config
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]) as mock_super:
            strategy.configure_fit(1, MagicMock(), MagicMock())
        mock_super.assert_called_once()

    def test_configure_fit_calls_on_round_start_callback(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        callback = MagicMock()
        strategy.on_round_start = callback
        loader.write({"proximal_mu": 0.01, "stop": False})
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(3, MagicMock(), MagicMock())
        callback.assert_called_once_with(3, {"proximal_mu": pytest.approx(0.01), "stop": False})

    def test_configure_fit_callback_exception_does_not_propagate(self, strategy_with_file_loader):
        strategy, loader = strategy_with_file_loader
        strategy.on_round_start = MagicMock(side_effect=RuntimeError("callback falhou"))
        with patch("flwr.server.strategy.FedProx.configure_fit", return_value=[]):
            strategy.configure_fit(1, MagicMock(), MagicMock())  # não deve lançar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
