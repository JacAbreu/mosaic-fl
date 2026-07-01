"""lifecycle_mixin.py — Ciclo de vida do processo: daemon (APScheduler), cron (run_once), heartbeat e parada segura."""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD, SCHEDULER_TIMEZONE

logger = logging.getLogger(__name__)


class _LifecycleMixin:
    """Requer os atributos definidos em FederatedScheduler.__init__ (interval_hours,
    min_clients, max_rounds, state, scheduler, _should_stop) e o método _job_round (de core.py)."""

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
        """Executa UM ciclo do scheduler (útil para cron) e empurra métricas ao Pushgateway."""
        logger.info("Modo run_once: executando um único ciclo.")
        self._job_round()

        try:
            from infrastructure.shared.metrics import push_metrics
            push_metrics(job="mosaicfl-scheduler")
        except Exception:
            pass
