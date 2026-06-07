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
from pathlib import Path
from typing import List, Optional

CONVERGENCE_THRESHOLD = 0.005
CONVERGENCE_PATIENCE = 3

logger = logging.getLogger(__name__)


class RoundDispatcher:
    """
    Dispara uma rodada de treinamento federado e coleta métricas.

    Nota: O Flower gerencia rounds quando clientes se conectam.
    Este dispatcher atua como supervisor externo que monitora métricas
    escritas pelo servidor em arquivo compartilhado.

    TODO produção: substituir _poll_round_metrics por chamada gRPC
    ou polling de endpoint REST /metrics no servidor Flower.
    """

    def __init__(self, server_address: str = "localhost:8080"):
        self.server_address = server_address

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
        """Polling por métricas da rodada em arquivo compartilhado."""
        metrics_file = Path(f"logs/round_{round_num}_metrics.json")

        for attempt in range(max_wait // 10):
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.debug(
                        "metrics_read_error",
                        extra={"round": round_num, "attempt": attempt, "error": str(e)},
                    )
            time.sleep(10)

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
