import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.mosaicfl_server.config_loader import ChromaDBConfigLoader


class TestChromaDBConfigLoader:
    """
    _cast converte os valores de metadata do ChromaDB (sempre strings ou primitivos)
    para os tipos corretos em Python. É a lógica mais crítica do loader — erros aqui
    causam stop=False quando deveria ser True, ou proximal_mu silenciosamente ignorado.
    """

    # ── _cast ─────────────────────────────────────────────────────────────────

    def test_cast_stop_true_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "true"})["stop"] is True

    def test_cast_stop_True_capitalized(self):
        assert ChromaDBConfigLoader._cast({"stop": "True"})["stop"] is True

    def test_cast_stop_one_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "1"})["stop"] is True

    def test_cast_stop_yes_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "yes"})["stop"] is True

    def test_cast_stop_false_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "false"})["stop"] is False

    def test_cast_stop_zero_string(self):
        assert ChromaDBConfigLoader._cast({"stop": "0"})["stop"] is False

    def test_cast_stop_native_bool_true(self):
        assert ChromaDBConfigLoader._cast({"stop": True})["stop"] is True

    def test_cast_stop_native_bool_false(self):
        assert ChromaDBConfigLoader._cast({"stop": False})["stop"] is False

    def test_cast_proximal_mu_string_float(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": "0.005"})
        assert result["proximal_mu"] == pytest.approx(0.005)

    def test_cast_proximal_mu_native_float(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": 0.01})
        assert result["proximal_mu"] == pytest.approx(0.01)

    def test_cast_proximal_mu_invalid_string_omitted(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": "nao_e_numero"})
        assert "proximal_mu" not in result

    def test_cast_proximal_mu_none_omitted(self):
        result = ChromaDBConfigLoader._cast({"proximal_mu": None})
        assert "proximal_mu" not in result

    def test_cast_pause_seconds_string_float(self):
        result = ChromaDBConfigLoader._cast({"pause_seconds": "30.5"})
        assert result["pause_seconds"] == pytest.approx(30.5)

    def test_cast_pause_seconds_invalid_string_omitted(self):
        result = ChromaDBConfigLoader._cast({"pause_seconds": "abc"})
        assert "pause_seconds" not in result

    def test_cast_unknown_key_passes_through(self):
        result = ChromaDBConfigLoader._cast({"custom_key": "valor"})
        assert result["custom_key"] == "valor"

    def test_cast_empty_dict_returns_empty(self):
        assert ChromaDBConfigLoader._cast({}) == {}

    def test_cast_mixed_keys(self):
        raw = {"stop": "true", "proximal_mu": "0.01", "pause_seconds": "5.0", "note": "test"}
        result = ChromaDBConfigLoader._cast(raw)
        assert result["stop"] is True
        assert result["proximal_mu"] == pytest.approx(0.01)
        assert result["pause_seconds"] == pytest.approx(5.0)
        assert result["note"] == "test"

    # ── load ──────────────────────────────────────────────────────────────────

    def _make_loader(self, metadata=None, raise_on_get=False):
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

    def test_load_empty_collection_returns_empty_dict(self):
        loader, _ = self._make_loader(metadata=None)
        assert loader.load(round_num=1) == {}

    def test_load_no_metadatas_key_returns_empty_dict(self):
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

    # ── write ─────────────────────────────────────────────────────────────────

    def _make_write_loader(self):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        with patch("chromadb.PersistentClient", return_value=mock_client):
            loader = ChromaDBConfigLoader(db_path="/fake/path")
        return loader, mock_collection

    def test_write_calls_upsert(self):
        loader, mock_collection = self._make_write_loader()
        loader.write({"proximal_mu": 0.01, "stop": False})
        mock_collection.upsert.assert_called_once()

    def test_write_uses_fixed_doc_id(self):
        loader, mock_collection = self._make_write_loader()
        loader.write({"stop": True})
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["runtime_config"]

    def test_write_filters_non_primitive_types(self):
        loader, mock_collection = self._make_write_loader()
        loader.write({"proximal_mu": 0.01, "nested": {"a": 1}, "lista": [1, 2]})
        call_kwargs = mock_collection.upsert.call_args[1]
        metadata = call_kwargs["metadatas"][0]
        assert "nested" not in metadata
        assert "lista" not in metadata
        assert "proximal_mu" in metadata

    def test_write_accepts_bool_values(self):
        loader, mock_collection = self._make_write_loader()
        loader.write({"stop": True})
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["metadatas"][0]["stop"] is True

    def test_write_exception_does_not_propagate(self):
        loader, mock_collection = self._make_write_loader()
        mock_collection.upsert.side_effect = Exception("falha de escrita")
        loader.write({"stop": True})

    # ── clear ─────────────────────────────────────────────────────────────────

    def test_clear_calls_delete_with_correct_id(self):
        loader, mock_collection = self._make_write_loader()
        loader.clear()
        mock_collection.delete.assert_called_once_with(ids=["runtime_config"])

    def test_clear_exception_does_not_propagate(self):
        loader, mock_collection = self._make_write_loader()
        mock_collection.delete.side_effect = Exception("falha")
        loader.clear()
