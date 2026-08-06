"""
Testes para EarlyStoppingServer (achado 2026-08-01) — cobre só a parte que
flwr.server.server.Server não oferece: parar o loop principal antes de
num_rounds quando a estratégia agenda um round-alvo
(strategy._early_stop_target_round). fit_round/evaluate_round/
_get_initial_parameters continuam sendo os métodos herdados de Server, sem
alteração — mockados aqui pra isolar só a lógica do loop.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flwr.common import Parameters
from flwr.server.client_manager import SimpleClientManager

from infrastructure.mosaicfl_server.runner.early_stop_server import EarlyStoppingServer


def _make_server(strategy):
    server = EarlyStoppingServer(client_manager=SimpleClientManager(), strategy=strategy)
    server._get_initial_parameters = MagicMock(
        return_value=Parameters(tensors=[], tensor_type="numpy.ndarray")
    )
    # fit_round/evaluate_round pertencem à classe base (Server) e não são
    # reescritos por EarlyStoppingServer — mockados só pra rodar sem SuperNodes
    # de verdade. strategy.evaluate (avaliação centralizada) retorna None no
    # Caminho B (sem test_loader central, por design de privacidade).
    server.fit_round = MagicMock(return_value=None)
    server.evaluate_round = MagicMock(return_value=None)
    strategy.evaluate = MagicMock(return_value=None)
    return server


class TestEarlyStoppingServerRunsAllRoundsWhenNoTarget:
    def test_runs_full_num_rounds_when_target_never_set(self):
        """Sem _early_stop_target_round (comportamento default, FL_EARLY_STOP
        desligado), o loop deve rodar até num_rounds — igual ao Server.fit()
        padrão do flwr, nenhuma regressão."""
        strategy = MagicMock()
        strategy._early_stop_target_round = None
        server = _make_server(strategy)

        server.fit(num_rounds=5, timeout=None)

        assert server.evaluate_round.call_count == 5
        called_rounds = [c.kwargs["server_round"] for c in server.evaluate_round.call_args_list]
        assert called_rounds == [1, 2, 3, 4, 5]


class TestEarlyStoppingServerBreaksOnTarget:
    def test_stops_at_target_round_before_ceiling(self):
        """Cenário do treino 83: teto=110, mas a estratégia agenda o alvo=37
        assim que evaluate_round roda a rodada 37 (efeito colateral do mock,
        simulando aggregate_evaluate já ter agendado o alvo numa rodada
        anterior) — o loop deve parar em 37, nunca chegar em 38."""
        strategy = MagicMock()
        strategy._early_stop_target_round = None

        def _evaluate_round_side_effect(server_round, timeout):
            if server_round == 36:
                strategy._early_stop_target_round = 37
            return None

        server = _make_server(strategy)
        server.evaluate_round.side_effect = _evaluate_round_side_effect

        server.fit(num_rounds=110, timeout=None)

        called_rounds = [c.kwargs["server_round"] for c in server.evaluate_round.call_args_list]
        assert called_rounds == list(range(1, 38))  # parou em 37, não foi até 110
        assert 38 not in called_rounds

    def test_target_already_reached_before_loop_starts_stops_on_round_one(self):
        """Caso degenerado (não deveria acontecer em produção, já que o alvo só
        é setado DENTRO de aggregate_evaluate de uma rodada que já rodou) — mas
        se o alvo já existir antes da rodada 1, o loop não deve rodar rodadas
        além dela."""
        strategy = MagicMock()
        strategy._early_stop_target_round = 1
        server = _make_server(strategy)

        server.fit(num_rounds=110, timeout=None)

        assert server.evaluate_round.call_count == 1
