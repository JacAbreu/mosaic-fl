"""
metrics.py — Registro Prometheus central do MOSAIC-FL.

Registry isolado (não usa o global do prometheus_client) para evitar colisões
entre testes e entre múltiplos processos que importem este módulo.

Uso pelo servidor:
    from infrastructure.shared.metrics import REGISTRY, fl_rounds_total, fl_round_accuracy
    fl_rounds_total.inc()
    fl_round_accuracy.set(0.87)

Uso pelo scheduler (CronJob — push antes de sair):
    from infrastructure.shared.metrics import push_metrics
    push_metrics(job="scheduler")
"""
import logging
import os

from prometheus_client import CollectorRegistry, Counter, Gauge

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()

fl_rounds_total = Counter(
    "fl_rounds_total",
    "Total de rounds de treinamento federado completados",
    registry=REGISTRY,
)

fl_round_accuracy = Gauge(
    "fl_round_accuracy",
    "Accuracy do último round completado",
    registry=REGISTRY,
)

fl_round_loss = Gauge(
    "fl_round_loss",
    "Loss do último round completado",
    registry=REGISTRY,
)

fl_clients_active = Gauge(
    "fl_clients_active",
    "Número de clientes que participaram do último round",
    registry=REGISTRY,
)

fl_convergence_round = Gauge(
    "fl_convergence_round",
    "Round em que convergência foi detectada (-1 = não convergiu ainda)",
    registry=REGISTRY,
)

# Inicializa convergence_round com -1 para indicar "não convergiu"
fl_convergence_round.set(-1)


def push_metrics(job: str, pushgateway_url: str | None = None) -> None:
    """
    Envia métricas ao Pushgateway (para jobs de curta duração como CronJob).

    Lê FL_PUSHGATEWAY_URL do ambiente se pushgateway_url não for fornecido.
    Falha silenciosamente com log de warning para não bloquear o processo principal.
    """
    url = pushgateway_url or os.getenv("FL_PUSHGATEWAY_URL")
    if not url:
        logger.debug("push_metrics_skipped: FL_PUSHGATEWAY_URL não definido")
        return

    try:
        from prometheus_client import push_to_gateway
        push_to_gateway(url, job=job, registry=REGISTRY)
        logger.info("metrics_pushed", extra={"job": job, "gateway": url})
    except Exception as exc:
        logger.warning("metrics_push_failed", extra={"job": job, "gateway": url, "error": str(exc)})
