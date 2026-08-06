"""early_stop_server.py — Server customizado com saída antecipada real na convergência.

Achado 2026-08-01, avaliando o treino 83 (DP uniforme): convergência foi detectada
na rodada 36 (pico, F1 macro=0,224), mas o treino seguiu até a 110 configurada —
entre as duas, o modelo colapsou (F1 macro=0,025) enquanto dp_epsilon_simple
cresceu de ~10 pra 1065 sem nenhum ganho de qualidade nesse intervalo. Relendo o
fonte instalado do flwr (1.32.1, flwr/server/server.py::Server.fit()), o loop
principal é `for current_round in range(1, num_rounds + 1)` — fixo, sem nenhuma
checagem de should_stop/break (confirmado de novo, mesmo achado de 2026-07-27,
ver ProductionFedProxStrategy.should_stop). O Caminho A sempre teve saída
antecipada real (experiments/training/core/fl_core/manual_loop.py:316-319) — é a
própria intenção documentada em FedConfig.num_rounds ("teto máximo; early
stopping pode parar antes"); o Caminho B nunca teve o equivalente porque herda
esse loop rígido sem alternativa.

`ServerAppComponents` (flwr/compat/server/serverapp_components.py) aceita um
`server: Server` customizado — esta subclasse copia o corpo de `Server.fit()`
(inalterado, mesma lógica de fit_round/evaluate_round herdada) e acrescenta a
única checagem que falta. Só é usada pelo SuperLink quando
`FED_CFG.early_stop` está ligado (FL_EARLY_STOP=true) — ver superlink.py.
"""
import timeit
from logging import INFO

from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.server import Server


class EarlyStoppingServer(Server):
    """Mesmo comportamento de `Server.fit()`, mas para de verdade quando a
    estratégia agenda um round-alvo de parada (`strategy._early_stop_target_round`,
    ver ProductionFedProxStrategy.aggregate_evaluate/is_final_round)."""

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """Run federated averaging for a number of rounds, com saída antecipada."""
        history = History()

        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            # Único trecho que não existe no Server.fit() original do flwr — todo
            # o resto acima é cópia fiel (ver docstring do módulo pra por quê).
            # evaluate_round() já rodou aggregate_evaluate() da estratégia pra esta
            # rodada, então _early_stop_target_round (se agendado) já reflete a
            # decisão tomada com os dados desta própria rodada.
            target_round = getattr(self.strategy, "_early_stop_target_round", None)
            if target_round is not None and current_round >= target_round:
                log(
                    INFO,
                    "early_stop_triggered round=%s target_round=%s (teto configurado=%s)",
                    current_round, target_round, num_rounds,
                )
                break

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed
