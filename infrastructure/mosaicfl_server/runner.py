"""
runner.py
Orquestrador do servidor Flower de produção (mosaicfl.v2).

Entrypoint único: main() — usado por __main__.py e server_daemon.py.
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import flwr as fl
import torch

from mosaicfl.v2.config import FED_CFG, RUNTIME_CFG
from mosaicfl.v2.model_v2 import SimplifiedBEHRT
from mosaicfl.v2.server_v2 import get_evaluate_fn, weighted_average_accuracy, weighted_average_loss

from .config_loader import get_config_loader
from .strategy import ProductionFedProxStrategy
from infrastructure.logging_setup import setup_logging as _setup_logging
from infrastructure.health_server import HealthServer

SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
HEALTH_PORT = int(os.getenv("FL_HEALTH_PORT", "8081"))

logger = logging.getLogger(__name__)

_health = HealthServer(port=HEALTH_PORT)


def setup_logging() -> None:
    """Configura logging estruturado (JSON ou texto) via logging_setup central."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_file="server_daemon.log")


def write_health_status(status: str, round_num: int = 0, clients: int = 0):
    """Atualiza o endpoint /healthz e persiste status em arquivo."""
    state = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
        "connected_clients": clients,
        "address": SERVER_ADDRESS,
    }
    _health.set_status(**state)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_DIR / "server_health.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.debug("Erro health check: %s", e)


class FederatedServer:
    """Servidor Flower de produção para MOSAIC-FL."""

    def __init__(
        self,
        address: str = SERVER_ADDRESS,
        num_rounds: int = FED_CFG.num_rounds,
        min_clients: int = FED_CFG.min_available_clients,
        proximal_mu: float = FED_CFG.proximal_mu,
    ):
        self.address = address
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.proximal_mu = proximal_mu
        self.global_model = self._init_model()
        self.test_loader = None
        self._shutdown_event = threading.Event()

    def _init_model(self) -> torch.nn.Module:
        model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        logger.info(
            "model_initialized",
            extra={"param_count": sum(p.numel() for p in model.parameters()), "device": str(RUNTIME_CFG.device)},
        )
        return model

    def _load_test_data(self) -> Optional[torch.utils.data.DataLoader]:
        """Carrega dados de teste holdout (opcional)."""
        return None

    def _signal_handler(self, signum, frame):
        logger.info("signal_received", extra={"signum": signum})
        self._shutdown_event.set()

    def _on_round_start(self, round_num: int, runtime_config: dict) -> None:
        """Callback chamado pela strategy antes de cada round."""
        write_health_status("running", round_num=round_num)
        if runtime_config:
            logger.debug("Round %s config: %s", round_num, runtime_config)

    def start(self):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _health.start()

        evaluate_fn = None
        self.test_loader = self._load_test_data()
        if self.test_loader is not None:
            evaluate_fn = get_evaluate_fn(self.test_loader)
            logger.info("evaluate_fn_enabled")

        config_loader = get_config_loader()

        strategy = ProductionFedProxStrategy(
            global_model=self.global_model,
            config_loader=config_loader,
            on_round_start=self._on_round_start,
            proximal_mu=self.proximal_mu,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=weighted_average_accuracy,
            fit_metrics_aggregation_fn=weighted_average_loss,
            on_fit_config_fn=lambda rnd: {"proximal_mu": self.proximal_mu, "round": rnd},
        )

        logger.info(
            "server_startup",
            extra={
                "address": self.address,
                "rounds": self.num_rounds,
                "min_clients": self.min_clients,
                "proximal_mu": self.proximal_mu,
                "device": str(RUNTIME_CFG.device),
                "checkpoint_dir": str(CHECKPOINT_DIR),
            },
        )

        write_health_status("starting")

        try:
            fl.server.start_server(
                server_address=self.address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
            )
        except Exception as e:
            logger.error("server_error", extra={"error": str(e)})
            write_health_status("error", clients=0)
            raise
        finally:
            write_health_status("stopped", round_num=strategy.round_counter)
            logger.info("server_stopped", extra={"rounds_completed": strategy.round_counter})


def main() -> None:
    """CLI do servidor de produção (mosaicfl.v2)."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Servidor Flower de produção — MOSAIC-FL (v2)",
    )
    parser.add_argument("--address", default=SERVER_ADDRESS, help="Endereço:porta")
    parser.add_argument("--port", type=int, default=8080, help="Porta")
    parser.add_argument(
        "--min-clients",
        type=int,
        default=FED_CFG.min_available_clients,
        help="Mínimo de clientes",
    )
    parser.add_argument("--rounds", type=int, default=FED_CFG.num_rounds, help="Máximo de rounds")
    parser.add_argument("--mu", type=float, default=FED_CFG.proximal_mu, help="Proximal mu")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(CHECKPOINT_DIR),
        help="Diretório de checkpoints",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FL_CHECKPOINT_DIR"] = str(checkpoint_dir)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["FL_LOG_DIR"] = str(LOG_DIR)

    if args.address == SERVER_ADDRESS:
        address = f"0.0.0.0:{args.port}"
    else:
        address = args.address

    FederatedServer(
        address=address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        proximal_mu=args.mu,
    ).start()
