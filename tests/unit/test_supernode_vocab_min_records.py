"""
Testes para FL_VOCAB_MIN_RECORDS em infrastructure/mosaicfl_client/runner/supernode.py
(_client_fn) — achado 2026-08-06: DataSource.find_vocab_candidates(vocab, min_records=100)
tinha o piso de volume hardcoded, nunca sobrescrito por nenhum chamador. Investigando dado
real do BPSP: com min_records=100, 145 candidatos aparecem mas NENHUM tem referência
institucional real (select_insertable rejeita todos os 145); baixando o piso pra <=94
(o de maior volume com referência real é PROTEINA_S_ATIVIDADE, 94 registros), pelo menos
1 candidato passa a ser aceito. FL_VOCAB_MIN_RECORDS torna isso configurável por treino,
sem mudar o default de produção (100, preservado quando a env var não é setada).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_client.runner.supernode import _client_fn


def _make_context(client_id="BPSP", data_source="sgbd"):
    ctx = MagicMock()
    ctx.node_config = {"client-id": client_id, "data-source": data_source}
    ctx.node_id = 12345
    return ctx


class TestVocabMinRecordsWiring:
    def test_default_preserved_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("FL_VOCAB_MIN_RECORDS", raising=False)
        source = MagicMock()
        source.find_vocab_candidates.return_value = []

        with patch(
            "infrastructure.mosaicfl_client.runner.supernode.DataSourceFactory.create",
            return_value=source,
        ), patch(
            "infrastructure.mosaicfl_client.runner.supernode.FedProxClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.to_client.return_value = "client-instance"
            _client_fn(_make_context())

        vocab_discovery_fn = mock_client_cls.call_args.kwargs["vocab_discovery_fn"]
        vocab_discovery_fn({"<PAD>": 0})
        source.find_vocab_candidates.assert_called_once_with({"<PAD>": 0}, min_records=100)

    def test_override_applied_when_env_set(self, monkeypatch):
        monkeypatch.setenv("FL_VOCAB_MIN_RECORDS", "90")
        source = MagicMock()
        source.find_vocab_candidates.return_value = []

        with patch(
            "infrastructure.mosaicfl_client.runner.supernode.DataSourceFactory.create",
            return_value=source,
        ), patch(
            "infrastructure.mosaicfl_client.runner.supernode.FedProxClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.to_client.return_value = "client-instance"
            _client_fn(_make_context())

        vocab_discovery_fn = mock_client_cls.call_args.kwargs["vocab_discovery_fn"]
        vocab_discovery_fn({"<PAD>": 0})
        source.find_vocab_candidates.assert_called_once_with({"<PAD>": 0}, min_records=90)

    def test_non_numeric_env_raises_clearly(self, monkeypatch):
        """Falha cedo e explícito (ValueError do int()) em vez de aceitar um valor
        inválido silenciosamente e só quebrar mais tarde, dentro do treino."""
        monkeypatch.setenv("FL_VOCAB_MIN_RECORDS", "not-a-number")
        source = MagicMock()

        with patch(
            "infrastructure.mosaicfl_client.runner.supernode.DataSourceFactory.create",
            return_value=source,
        ), pytest.raises(ValueError):
            _client_fn(_make_context())
