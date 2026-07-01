"""
core.py — FederatedScheduler: inicialização e o ciclo principal (_job_round).

SchedulerState, SchedulerStateStore, ClientAvailabilityChecker e RoundDispatcher
ficam importados aqui (não em módulos separados) porque os testes fazem patch
direto em "infrastructure.mosaicfl_scheduler.scheduler_daemon.core.X" — o
construtor que instancia essas classes precisa estar no mesmo módulo onde
elas são importadas para o patch ser efetivo.
"""
import logging
import socket
from datetime import datetime

from .config import MAX_ROUNDS, MIN_AVAILABLE_CLIENTS, SCHEDULER_INTERVAL_HOURS
from .lifecycle_mixin import _LifecycleMixin

logger = logging.getLogger(__name__)

# Imports dos módulos do scheduler usando imports absolutos
try:
    from ..schedule_state import SchedulerState
    from ..state_store import SchedulerStateStore
    from ..client_availability_checker import ClientAvailabilityChecker
    from ..round_training_fl_dispatcher import RoundDispatcher
except ImportError as e:
    import sys
    logger.error(f"Erro importando módulos do scheduler: {e}")
    logger.error("Certifique-se de que o pacote mosaicfl está instalado:")
    logger.error("  pip install -e .")
    logger.error("Ou configure o PYTHONPATH:")
    logger.error("  export PYTHONPATH=/home/jacabreu/studies/usp/tcc/mosaic-fl:$PYTHONPATH")
    sys.exit(1)


class FederatedScheduler(_LifecycleMixin):
    """
    Scheduler de rounds federados usando APScheduler.
    """

    def __init__(
        self,
        interval_hours: float = SCHEDULER_INTERVAL_HOURS,
        min_clients: int = MIN_AVAILABLE_CLIENTS,
        max_rounds: int = MAX_ROUNDS,
        server_address: str = "localhost:8080",
    ):
        self.interval_hours = interval_hours
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.server_address = server_address

        self.checker = ClientAvailabilityChecker()
        self.dispatcher = RoundDispatcher(server_address=server_address)
        self._store = SchedulerStateStore()
        self.state = self._store.load()
        self.scheduler = None
        self._should_stop = False  # flag para controle de parada

    def _check_server_connectivity(self) -> bool:
        """
        Verifica se o servidor Flower está acessível.

        Retorna True se o servidor responde, False caso contrário.
        """
        try:
            host, port = self.server_address.split(":")
            port = int(port)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # Timeout de 5 segundos
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                logger.debug(f"Servidor Flower acessível em {self.server_address}")
                return True
            else:
                logger.warning(f"Servidor Flower NÃO está acessível em {self.server_address}")
                return False

        except Exception as e:
            logger.error(f"Erro ao verificar conectividade com servidor: {e}")
            return False

    def _job_round(self):
        """Job executado a cada ciclo do scheduler."""
        logger.info("=" * 60)
        logger.info(f"SCHEDULER — Ciclo iniciado às {datetime.now().isoformat()}")
        logger.info("=" * 60)

        # VERIFICACAO DE CONECTIVIDADE COM O SERVIDOR
        if not self._check_server_connectivity():
            logger.error("SERVIDOR FLOWER NAO ESTA ACESSIVEL!")
            logger.error(f"   Endereço configurado: {self.server_address}")
            logger.error("")
            logger.error("Possíveis causas:")
            logger.error("  1. O servidor Flower não foi iniciado")
            logger.error("  2. O servidor está em outro host/porta")
            logger.error("  3. Firewall bloqueando a conexão")
            logger.error("")
            logger.error("Ações recomendadas:")
            logger.error("  • Inicie o servidor: python infrastructure/server/server_daemon.py")
            logger.error("  • Verifique a variável FL_SERVER_ADDRESS")
            logger.error("  • Verifique se a porta está aberta: nc -zv localhost 8080")
            logger.error("")
            logger.warning("O scheduler continuará tentando no próximo ciclo...")
            return

        # 1. Verifica se já convergiu ou atingiu limite
        if self.state.converged:
            logger.info("Convergência já atingida. Scheduler dormindo indefinidamente.")
            self._stop_scheduler()
            return

        if self.state.total_rounds_completed >= self.max_rounds:
            logger.info(f"Máximo de {self.max_rounds} rounds atingido. Parando.")
            self._stop_scheduler()
            return

        # 2. Verifica clientes disponíveis
        num_clients, active_clients = self.checker.check_via_server()
        logger.info(f"Clientes disponíveis: {num_clients}/{self.min_clients} necessários")
        logger.info(f"Clientes ativos: {active_clients}")

        try:
            from infrastructure.shared.metrics import fl_clients_active
            fl_clients_active.set(num_clients)
        except Exception:
            pass

        if num_clients < self.min_clients:
            logger.warning(
                f"Clientes insuficientes ({num_clients} < {self.min_clients}). "
                f"Aguardando próximo ciclo em {self.interval_hours}h."
            )
            return

        # 3. Dispara round — retorna accuracy ou None em caso de falha
        next_round = self.state.total_rounds_completed + 1
        accuracy = self.dispatcher.dispatch_round(next_round, active_clients)

        if accuracy is not None:
            self.state.total_rounds_completed += 1
            self.state.current_round = next_round
            self.state.last_run = datetime.now().isoformat()
            self.state.accuracy_history.append(accuracy)

            # 4. Verifica convergência com o histórico atualizado
            converged = self.dispatcher.check_convergence(self.state.accuracy_history)
            self.state.converged = converged
            if converged:
                self.state.convergence_round = next_round

            self._store.save(self.state)
            self._store.record_round(next_round, accuracy=accuracy, success=True)

            if converged:
                logger.info("Convergência detectada. Scheduler será parado.")
                self._stop_scheduler()
        else:
            self._store.record_round(next_round, accuracy=None, success=False)
            logger.error(f"Falha ao executar round {next_round}.")
