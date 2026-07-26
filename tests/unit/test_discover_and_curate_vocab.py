"""
Testes para ProductionFedProxStrategy._discover_and_curate_vocab() e
initialize_parameters() — achado 2026-07-26, desenho do vocabulário federado
bidirecional. Roda uma única vez, antes da Rodada 1 (ver
strategy/core.py::initialize_parameters).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(vocab=None, min_available_clients=2):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy.vocab = vocab if vocab is not None else {"CLS": 0}
    strategy.min_available_clients = min_available_clients
    return strategy


def _client_proxy(candidates):
    proxy = MagicMock()
    proxy.evaluate.return_value = SimpleNamespace(
        metrics={"vocab_candidates_json": json.dumps(candidates)}
    )
    return proxy


def _client_manager(clients, wait_for_result=True):
    cm = MagicMock()
    cm.wait_for.return_value = wait_for_result
    cm.all.return_value = {i: c for i, c in enumerate(clients)}
    return cm


class TestDiscoverAndCurateVocab:
    def test_waits_for_min_available_clients(self):
        strategy = _make_strategy(min_available_clients=3)
        cm = _client_manager([], wait_for_result=True)

        strategy._discover_and_curate_vocab(cm)

        cm.wait_for.assert_called_once()
        assert cm.wait_for.call_args.args[0] == 3

    def test_timeout_keeps_vocab_unchanged(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([_client_proxy([])], wait_for_result=False)

        strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == {"CLS": 0}
        cm.all.assert_not_called()

    def test_no_clients_keeps_vocab_unchanged(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([], wait_for_result=True)

        strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == {"CLS": 0}

    def test_merges_candidates_across_clients(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        client_a = _client_proxy([{"analyte": "PCR", "n_records": 100, "has_real_ref": False}])
        client_b = _client_proxy([{"analyte": "PCR", "n_records": 50, "has_real_ref": True}])
        cm = _client_manager([client_a, client_b])

        captured = {}

        def _fake_select_insertable(candidates, **kwargs):
            captured["candidates"] = candidates
            return []

        with patch("scripts.discover_analyte_catalog_gaps.select_insertable", side_effect=_fake_select_insertable), \
             patch("scripts.discover_analyte_catalog_gaps.insert_candidates"), \
             patch("scripts.build_standard_vocab.build_standard_vocab"):
            strategy._discover_and_curate_vocab(cm)

        merged = captured["candidates"]
        assert len(merged) == 1
        assert merged[0]["analyte"] == "PCR"
        assert merged[0]["n_records"] == 150
        assert merged[0]["n_hospitals"] == 2
        assert merged[0]["has_real_ref"] is True

    def test_no_candidates_from_any_client_keeps_vocab(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([_client_proxy([]), _client_proxy([])])

        strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == {"CLS": 0}

    def test_no_selected_candidates_keeps_vocab(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([_client_proxy([{"analyte": "RARO", "n_records": 1, "has_real_ref": False}])])

        with patch("scripts.discover_analyte_catalog_gaps.select_insertable", return_value=[]):
            strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == {"CLS": 0}

    def test_updates_vocab_when_candidates_selected_and_inserted(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([_client_proxy([{"analyte": "PCR", "n_records": 200, "has_real_ref": True}])])
        new_vocab = {"CLS": 0, "PCR_HIGH": 1, "PCR_NORMAL": 2, "PCR_LOW": 3}

        with patch("scripts.discover_analyte_catalog_gaps.select_insertable",
                   return_value=[{"analyte": "PCR", "canonical": "PCR"}]), \
             patch("scripts.discover_analyte_catalog_gaps.insert_candidates", return_value=1), \
             patch("scripts.build_standard_vocab.build_standard_vocab", return_value=new_vocab), \
             patch("sqlalchemy.create_engine"):
            strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == new_vocab

    def test_vocab_overflow_keeps_previous_vocab(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([_client_proxy([{"analyte": "PCR", "n_records": 200, "has_real_ref": True}])])
        oversized_vocab = {f"T{i}": i for i in range(100000)}

        with patch("scripts.discover_analyte_catalog_gaps.select_insertable",
                   return_value=[{"analyte": "PCR", "canonical": "PCR"}]), \
             patch("scripts.discover_analyte_catalog_gaps.insert_candidates", return_value=1), \
             patch("scripts.build_standard_vocab.build_standard_vocab", return_value=oversized_vocab), \
             patch("sqlalchemy.create_engine"):
            strategy._discover_and_curate_vocab(cm)

        assert strategy.vocab == {"CLS": 0}

    def test_client_evaluate_exception_does_not_propagate(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        bad_client = MagicMock()
        bad_client.evaluate.side_effect = RuntimeError("cliente fora do ar")
        cm = _client_manager([bad_client])

        strategy._discover_and_curate_vocab(cm)  # não deve levantar

        assert strategy.vocab == {"CLS": 0}

    def test_unexpected_exception_never_propagates(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = MagicMock()
        cm.wait_for.side_effect = RuntimeError("erro inesperado")

        strategy._discover_and_curate_vocab(cm)  # não deve levantar

        assert strategy.vocab == {"CLS": 0}

    def test_initialize_parameters_calls_discovery_then_super(self):
        strategy = _make_strategy(vocab={"CLS": 0})
        cm = _client_manager([], wait_for_result=True)
        sentinel = object()

        with patch.object(strategy, "_discover_and_curate_vocab") as mock_discover, \
             patch("flwr.server.strategy.FedProx.initialize_parameters", return_value=sentinel) as mock_super:
            result = strategy.initialize_parameters(cm)

        mock_discover.assert_called_once_with(cm)
        mock_super.assert_called_once_with(cm)
        assert result is sentinel
