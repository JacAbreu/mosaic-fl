"""
Testes para _FitConfigMixin._inject_current_vocab() / configure_fit() /
configure_evaluate() — achado 2026-07-26, desenho do vocabulário federado
bidirecional. self.vocab (atualizado uma única vez em initialize_parameters(),
antes da Rodada 1 — ver strategy/core.py::_discover_and_curate_vocab) precisa
sobrescrever qualquer vocab_json que os lambdas on_fit_config_fn/
on_evaluate_config_fn (superlink.py) tenham calculado por closure.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.mosaicfl_server.strategy.fit_config_mixin import _FitConfigMixin


class _FakeBaseStrategy:
    """Substitui fl.server.strategy.FedProx — devolve instructions fixas com um
    vocab_json 'antigo' (simulando o que os lambdas de superlink.py calculariam
    por closure), pra confirmar que o mixin sobrescreve com self.vocab."""

    def configure_fit(self, server_round, parameters, client_manager):
        ins = SimpleNamespace(config={"vocab_json": "OLD_FIT_CLOSURE_VALUE", "round": server_round})
        return [(MagicMock(), ins)]

    def configure_evaluate(self, server_round, parameters, client_manager):
        ins = SimpleNamespace(config={"vocab_json": "OLD_EVAL_CLOSURE_VALUE", "round": server_round})
        return [(MagicMock(), ins)]


class _StrategyUnderTest(_FitConfigMixin, _FakeBaseStrategy):
    def __init__(self, vocab):
        self.vocab = vocab
        self.config_loader = MagicMock()
        self.config_loader.load.return_value = {}
        self.on_round_start = None
        self.proximal_mu = 0.01
        self.should_stop = False

    def _start_round_watchdog(self, server_round):
        pass


class TestInjectCurrentVocab:
    def test_configure_fit_overwrites_closure_vocab_with_self_vocab(self):
        strategy = _StrategyUnderTest(vocab={"CLS": 0, "PCR_HIGH": 1})

        instructions = strategy.configure_fit(1, None, MagicMock())

        _, ins = instructions[0]
        assert json_loads_vocab(ins.config["vocab_json"]) == {"CLS": 0, "PCR_HIGH": 1}

    def test_configure_evaluate_overwrites_closure_vocab_with_self_vocab(self):
        strategy = _StrategyUnderTest(vocab={"CLS": 0, "PCR_HIGH": 1})

        instructions = strategy.configure_evaluate(1, None, MagicMock())

        _, ins = instructions[0]
        assert json_loads_vocab(ins.config["vocab_json"]) == {"CLS": 0, "PCR_HIGH": 1}

    def test_reflects_vocab_updated_after_discovery(self):
        """self.vocab pode ter sido atualizado por _discover_and_curate_vocab()
        entre a criação da estratégia e a 1ª chamada de configure_fit — o mixin
        precisa ler self.vocab no momento da chamada, não um valor capturado antes."""
        strategy = _StrategyUnderTest(vocab={"CLS": 0})
        strategy.vocab = {"CLS": 0, "NOVO_ANALITO_HIGH": 5}  # simula pós-descoberta

        instructions = strategy.configure_fit(1, None, MagicMock())

        _, ins = instructions[0]
        assert json_loads_vocab(ins.config["vocab_json"]) == {"CLS": 0, "NOVO_ANALITO_HIGH": 5}

    def test_configure_fit_preserves_other_config_keys(self):
        strategy = _StrategyUnderTest(vocab={"CLS": 0})

        instructions = strategy.configure_fit(7, None, MagicMock())

        _, ins = instructions[0]
        assert ins.config["round"] == 7

    def test_stop_short_circuits_before_vocab_injection(self):
        strategy = _StrategyUnderTest(vocab={"CLS": 0})
        strategy.config_loader.load.return_value = {"stop": True}

        instructions = strategy.configure_fit(1, None, MagicMock())

        assert instructions == []
        assert strategy.should_stop is True


def json_loads_vocab(raw):
    import json
    return json.loads(raw)
