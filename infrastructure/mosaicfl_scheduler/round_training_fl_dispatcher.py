"""
round_dispatcher.py
Dispara e monitora uma rodada de treinamento federado.

O dispatcher não gerencia estado — apenas coleta métricas e verifica
convergência. Toda a persistência é responsabilidade do FederatedScheduler
via SchedulerStateStore.
"""
import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

CONVERGENCE_THRESHOLD = 0.005
CONVERGENCE_PATIENCE = 3

logger = logging.getLogger(__name__)


class RoundDispatcher:
    """
    Dispara uma rodada de treinamento federado e coleta métricas.

    Consulta métricas via HTTP GET no HealthServer do servidor Flower
    (endpoint /metrics/round/{n}), usando backoff exponencial até o
    round completar ou o timeout ser atingido.

    O endereço de saúde é derivado automaticamente de server_address
    (mesmo host, porta 8081), podendo ser sobrescrito por health_address.
    """

    def __init__(
        self,
        server_address: str = "localhost:8080",
        health_address: Optional[str] = None,
    ):
        self.server_address = server_address
        if health_address is None:
            host = server_address.split(":")[0]
            health_address = f"http://{host}:8081"
        self.health_address = health_address

    def dispatch_round(self, round_num: int, active_clients: List[str]) -> Optional[float]:
        """
        Monitora a execução de um round e retorna a accuracy coletada.

        Returns:
            accuracy (float) se as métricas foram coletadas com sucesso,
            None se o round falhou ou métricas não estavam disponíveis.
        """
        logger.info(
            "round_dispatched",
            extra={"round": round_num, "client_count": len(active_clients), "clients": active_clients},
        )

        metrics = self._poll_round_metrics(round_num)

        if metrics:
            accuracy = metrics.get("accuracy")
            logger.info(
                "round_completed",
                extra={"round": round_num, "accuracy": accuracy, "loss": metrics.get("loss")},
            )
            return accuracy

        logger.warning("round_metrics_unavailable", extra={"round": round_num})
        return None

    def _poll_round_metrics(self, round_num: int, max_wait: int = 600) -> Optional[dict]:
        """
        Consulta métricas do round via HTTP GET com backoff exponencial.

        Faz GET {health_address}/metrics/round/{round_num}:
          - 200 → métricas disponíveis, retorna o dict
          - 404 → round ainda não concluído, aguarda e tenta de novo
          - erro de conexão → servidor ainda subindo, aguarda e tenta de novo
        """
        url = f"{self.health_address}/metrics/round/{round_num}"
        delay = 5
        elapsed = 0

        while elapsed < max_wait:
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    if resp.status == 200:
                        body = resp.read().decode("utf-8")
                        metrics = json.loads(body)
                        logger.debug(
                            "metrics_fetched",
                            extra={"round": round_num, "elapsed": elapsed},
                        )
                        return metrics
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    logger.debug(
                        "metrics_not_ready",
                        extra={"round": round_num, "elapsed": elapsed, "next_check": delay},
                    )
                else:
                    logger.warning(
                        "metrics_http_error",
                        extra={"round": round_num, "status": exc.code, "error": str(exc)},
                    )
            except Exception as exc:
                logger.debug(
                    "metrics_connection_error",
                    extra={"round": round_num, "elapsed": elapsed, "error": str(exc)},
                )

            time.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, 60)

        logger.warning(
            "metrics_timeout",
            extra={"round": round_num, "max_wait": max_wait},
        )
        return None

    def check_convergence(self, accuracy_history: List[float]) -> bool:
        """
        Verifica se convergência foi atingida com base no histórico fornecido.

        Args:
            accuracy_history: histórico completo de accuracy, mantido pelo caller.
        """
        if len(accuracy_history) < CONVERGENCE_PATIENCE + 1:
            return False

        recent = accuracy_history[-(CONVERGENCE_PATIENCE + 1):]
        deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return all(d < CONVERGENCE_THRESHOLD for d in deltas)
