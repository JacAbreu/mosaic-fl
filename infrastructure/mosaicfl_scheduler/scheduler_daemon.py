"""
scheduler_daemon.py
Scheduler de rounds federados usando APScheduler.

Padrão de execução:
  1. Desperta (trigger: intervalo ou horário específico)
  2. Verifica clientes disponíveis
  3. Se >= min_available_clients: dispara round
  4. Se < min_available_clients: loga e volta a dormir
  5. Verifica convergência
  6. Se convergiu ou max_rounds: para agendamento

Uso:
    python scheduler_daemon.py              # modo daemon (loop infinito)
    python scheduler_daemon.py --once       # modo cron (1 execução e termina)
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Configuração de logging (único ponto de configuração)
LOG_FILE = Path(os.getenv("FL_SCHEDULER_LOG", "logs/scheduler.log"))
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Imports dos módulos do scheduler usando imports absolutos
try:
    from .schedule_state import SchedulerState
    from .client_availability_checker import ClientAvailabilityChecker
    from .round_training_fl_dispatcher import RoundDispatcher
except ImportError as e:
    logger.error(f"Erro importando módulos do scheduler: {e}")
    logger.error("Certifique-se de que o pacote mosaicfl está instalado:")
    logger.error("  pip install -e .")
    logger.error("Ou configure o PYTHONPATH:")
    logger.error("  export PYTHONPATH=/home/jacabreu/studies/usp/tcc/mosaic-fl:$PYTHONPATH")
    sys.exit(1)

# Configurações via variáveis de ambiente
SCHEDULER_INTERVAL_HOURS = float(os.getenv("FL_SCHEDULER_INTERVAL_HOURS", "6"))
SCHEDULER_TIMEZONE = os.getenv("FL_SCHEDULER_TIMEZONE", "America/Sao_Paulo")
MIN_AVAILABLE_CLIENTS = int(os.getenv("FL_SCHEDULER_MIN_CLIENTS", "3"))
MAX_ROUNDS = int(os.getenv("FL_SCHEDULER_MAX_ROUNDS", "20"))
CONVERGENCE_THRESHOLD = float(os.getenv("FL_SCHEDULER_CONV_THRESHOLD", "0.005"))
CONVERGENCE_PATIENCE = int(os.getenv("FL_SCHEDULER_CONV_PATIENCE", "3"))


class FederatedScheduler:
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
        self.state = SchedulerState.load()
        self.scheduler = None
        self._should_stop = False  # flag para controle de parada

    def _check_server_connectivity(self) -> bool:
        """
        Verifica se o servidor Flower está acessível.
        
        Retorna True se o servidor responde, False caso contrário.
        """
        try:
            import socket
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
        
        # ⚠️ VERIFICAÇÃO DE CONECTIVIDADE COM O SERVIDOR
        if not self._check_server_connectivity():
            logger.error("🚫 SERVIDOR FLOWER NÃO ESTÁ ACESSÍVEL!")
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

        if num_clients < self.min_clients:
            logger.warning(
                f"Clientes insuficientes ({num_clients} < {self.min_clients}). "
                f"Aguardando próximo ciclo em {self.interval_hours}h."
            )
            return

        # 3. Dispara round
        next_round = self.state.total_rounds_completed + 1
        success = self.dispatcher.dispatch_round(next_round, active_clients)

        if success:
            # 4. Verifica convergência
            converged = self.dispatcher.check_convergence()
            if converged:
                logger.info("Convergência detectada. Scheduler será parado.")
                self._stop_scheduler()
        else:
            logger.error(f"Falha ao executar round {next_round}.")

    def _stop_scheduler(self):
        """Para o scheduler de forma segura (pausa o job, não o processo)."""
        if self.scheduler:
            try:
                # Pausa o job específico (a API do scheduler não tem pause() global)
                self.scheduler.pause_job("federated_round")
                logger.info("Job 'federated_round' pausado.")
            except Exception as e:
                logger.warning(f"Erro ao pausar job: {e}")
        self._should_stop = True

    def _heartbeat(self):
        """Registra que o scheduler está ativo."""
        heartbeat_file = Path("logs/scheduler_heartbeat.json")
        heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(heartbeat_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "round": self.state.total_rounds_completed,
                        "converged": self.state.converged,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.debug(f"Erro ao escrever heartbeat: {e}")

    def start_daemon(self):
        """Inicia o scheduler como daemon (fica rodando indefinidamente)."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            logger.error("APScheduler não instalado. Execute: pip install apscheduler")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("MOSAIC-FL — SCHEDULER DE ROUNDS FEDERADOS")
        logger.info("=" * 60)
        logger.info(f"Intervalo:      a cada {self.interval_hours}h")
        logger.info(f"Min clientes:   {self.min_clients}")
        logger.info(f"Max rounds:     {self.max_rounds}")
        logger.info(f"Convergência:   Δ < {CONVERGENCE_THRESHOLD} por {CONVERGENCE_PATIENCE} rounds")
        logger.info(
            f"Estado atual:   round {self.state.total_rounds_completed}, "
            f"converged={self.state.converged}"
        )
        logger.info("=" * 60)

        self.scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)

        # Job principal: verifica e dispara rounds
        self.scheduler.add_job(
            self._job_round,
            trigger=IntervalTrigger(hours=self.interval_hours),
            id="federated_round",
            name="Federated Learning Round",
            replace_existing=True,
        )

        # Job de heartbeat: registra que scheduler está vivo
        self.scheduler.add_job(
            self._heartbeat,
            trigger=IntervalTrigger(minutes=5),
            id="scheduler_heartbeat",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("Scheduler iniciado. Pressione Ctrl+C para parar.")

        try:
            while not self._should_stop:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler interrompido pelo usuário.")
        finally:
            if self.scheduler:
                self.scheduler.shutdown()
                logger.info("Scheduler finalizado.")

    def run_once(self):
        """Executa UM ciclo do scheduler (útil para cron)."""
        logger.info("Modo run_once: executando um único ciclo.")
        self._job_round()


def main():
    parser = argparse.ArgumentParser(description="Scheduler de Rounds Federados MOSAIC-FL")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Executa um único ciclo e termina (para uso com system cron)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SCHEDULER_INTERVAL_HOURS,
        help=f"Intervalo entre ciclos em horas (default: {SCHEDULER_INTERVAL_HOURS})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=MIN_AVAILABLE_CLIENTS,
        help=f"Mínimo de clientes para iniciar round (default: {MIN_AVAILABLE_CLIENTS})",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS,
        help=f"Máximo de rounds (default: {MAX_ROUNDS})",
    )
    args = parser.parse_args()

    scheduler = FederatedScheduler(
        interval_hours=args.interval,
        min_clients=args.min_clients,
        max_rounds=args.max_rounds,
    )

    if args.once:
        scheduler.run_once()
    else:
        scheduler.start_daemon()


if __name__ == "__main__":
    main()