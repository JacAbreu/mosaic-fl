"""
round_dispatcher.py
Dispara e monitora uma rodada de treinamento federado.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Configurações podem vir de variáveis de ambiente ou config.py
CONVERGENCE_THRESHOLD = 0.005
CONVERGENCE_PATIENCE = 3

logger = logging.getLogger(__name__)


class RoundDispatcher:
    """
    Dispara uma rodada de treinamento federado e monitora métricas.

    Nota: O Flower nativamente gerencia rounds quando clientes se conectam.
    Este dispatcher atua como supervisor externo que:
      - Verifica se o servidor completou a rodada
      - Coleta métricas de arquivo compartilhado (ou endpoint REST)
      - Persiste estado e detecta convergência
    """

    def __init__(self, server_address: str = "localhost:8080"):
        self.server_address = server_address
        # Import absoluto do estado do scheduler
        from .schedule_state import SchedulerState
        self.state = SchedulerState.load()

    def dispatch_round(self, round_num: int, active_clients: List[str]) -> bool:
        """
        Registra início de rodada e aguarda métricas.

        Em produção com Flower:
          - O servidor já está rodando (server_daemon.py)
          - Clientes já estão conectados (client_daemon.py)
          - O servidor inicia automaticamente quando min_available_clients conectam
          - Este dispatcher apenas MONITORA e REGISTRA o resultado

        Retorna: True se métricas foram coletadas com sucesso.
        """
        logger.info(
            f"Dispatching round {round_num} com {len(active_clients)} clientes: {active_clients}"
        )

        # Em produção real, substituir por:
        #   - Chamada gRPC custom ao servidor para forçar início de round
        #   - Ou polling de endpoint REST /metrics
        #   - Ou leitura de fila (Redis, RabbitMQ)

        # Aguarda métricas via polling (não bloqueia com sleep fixo)
        logger.info("Aguardando métricas da rodada via polling (max 10 min)...")
        metrics = self._poll_round_metrics(round_num)

        if metrics:
            self.state.accuracy_history.append(metrics.get("accuracy", 0.0))
            self.state.total_rounds_completed = round_num
            self.state.last_run = datetime.now().isoformat()
            self.state.save()

            logger.info(f"Round {round_num} completado: accuracy={metrics.get('accuracy', 'N/A')}")
            return True
        else:
            logger.warning(f"Round {round_num}: métricas não disponíveis.")
            return False

    def _poll_round_metrics(self, round_num: int, max_wait: int = 600) -> Optional[dict]:
        """Polling por métricas da rodada em arquivo compartilhado."""
        metrics_file = Path(f"logs/round_{round_num}_metrics.json")

        for attempt in range(max_wait // 10):
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.debug(f"Erro lendo métricas (tentativa {attempt}): {e}")
            time.sleep(10)

        return None

    def check_convergence(self) -> bool:
        """Verifica se convergência foi atingida."""
        if len(self.state.accuracy_history) < CONVERGENCE_PATIENCE + 1:
            return False

        recent = self.state.accuracy_history[-(CONVERGENCE_PATIENCE + 1):]
        deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]

        converged = all(d < CONVERGENCE_THRESHOLD for d in deltas)

        if converged and not self.state.converged:
            self.state.converged = True
            self.state.convergence_round = self.state.total_rounds_completed
            self.state.save()
            logger.info(
                f"🎯 CONVERGÊNCIA ATINGIDA na rodada {self.state.convergence_round}!"
            )

        return converged