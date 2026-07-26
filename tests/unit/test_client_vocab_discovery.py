"""
Testes para o branch discover_vocab_only em FedProxClient.evaluate() — achado
2026-07-26, desenho do vocabulário federado bidirecional (ver
ProductionFedProxStrategy._discover_and_curate_vocab, roda antes da Rodada 1).
"""
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import VOCAB_SIZE, NUM_CLASSES


@pytest.fixture
def dummy_loader():
    x = torch.randint(1, VOCAB_SIZE, (8, 16))
    y = torch.randint(0, NUM_CLASSES, (8,))
    dia = torch.randint(0, 100, (8, 16))
    return DataLoader(TensorDataset(x, y, dia), batch_size=4)


def _unused_loader_factory(vocab_json):
    raise AssertionError("loader_factory não deveria ser chamado no branch discover_vocab_only")


class TestDiscoverVocabOnlyBranch:
    def test_returns_candidates_from_discovery_fn(self):
        candidates = [{"analyte": "PCR", "n_records": 200, "has_real_ref": False}]
        client = FedProxClient(
            client_id=0, loader_factory=_unused_loader_factory,
            vocab_discovery_fn=lambda vocab: candidates,
        )

        loss, n_samples, metrics = client.evaluate(
            [], {"discover_vocab_only": True, "vocab_json": json.dumps({"CLS": 1})}
        )

        assert loss == 0.0
        assert n_samples == 0
        assert json.loads(metrics["vocab_candidates_json"]) == candidates

    def test_passes_parsed_vocab_to_discovery_fn(self):
        received = {}

        def _fn(vocab):
            received.update(vocab)
            return []

        client = FedProxClient(client_id=0, loader_factory=_unused_loader_factory, vocab_discovery_fn=_fn)
        client.evaluate([], {"discover_vocab_only": True, "vocab_json": json.dumps({"PCR_HIGH": 3})})

        assert received == {"PCR_HIGH": 3}

    def test_no_discovery_fn_returns_empty_candidates(self):
        client = FedProxClient(client_id=0, loader_factory=_unused_loader_factory, vocab_discovery_fn=None)

        loss, n_samples, metrics = client.evaluate(
            [], {"discover_vocab_only": True, "vocab_json": json.dumps({})}
        )

        assert loss == 0.0
        assert n_samples == 0
        assert json.loads(metrics["vocab_candidates_json"]) == []

    def test_discovery_fn_exception_returns_empty_candidates_not_raise(self):
        def _boom(vocab):
            raise RuntimeError("conexão recusada")

        client = FedProxClient(client_id=0, loader_factory=_unused_loader_factory, vocab_discovery_fn=_boom)

        loss, n_samples, metrics = client.evaluate(
            [], {"discover_vocab_only": True, "vocab_json": json.dumps({})}
        )

        assert json.loads(metrics["vocab_candidates_json"]) == []

    def test_does_not_touch_model_or_data(self, dummy_loader, monkeypatch):
        """Branch sai antes de _ensure_data()/set_parameters() — não deve chamar
        nenhum dos dois, mesmo quando train/val_loader e parâmetros reais estão
        disponíveis."""
        client = FedProxClient(client_id=0, train_loader=dummy_loader, val_loader=dummy_loader)

        called = {"ensure_data": False, "set_parameters": False}
        monkeypatch.setattr(client, "_ensure_data", lambda cfg: called.__setitem__("ensure_data", True))
        monkeypatch.setattr(client, "set_parameters", lambda params: called.__setitem__("set_parameters", True))

        client.evaluate([], {"discover_vocab_only": True, "vocab_json": "{}"})

        assert called == {"ensure_data": False, "set_parameters": False}

    def test_normal_evaluate_unaffected_when_flag_absent(self, dummy_loader):
        client = FedProxClient(client_id=0, train_loader=dummy_loader, val_loader=dummy_loader)
        params = client.get_parameters({})

        loss, n_samples, metrics = client.evaluate(params, {})

        assert n_samples == len(dummy_loader.dataset)
        assert "vocab_candidates_json" not in metrics
