"""
runner.py
Orquestrador do servidor Flower de produção (mosaicfl.v2).

Dois modos de execução:

  SuperLink (produção):
      flower-superlink ...         # infraestrutura persistente
      flwr run . local             # dispara ServerApp via SuperLink
    Expõe: app = ServerApp(server_fn=_make_server_components)

  Legado (desenvolvimento local):
      python -m infrastructure.mosaicfl_server --port 8080
    Usa: FederatedServer + fl.server.start_server
"""
import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import flwr as fl
import torch
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from mosaicfl.core.config import FED_CFG, RUNTIME_CFG
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.federated import get_evaluate_fn, weighted_average_accuracy, weighted_average_loss

from .config_loader import get_config_loader
from .state_store import TrainingStateStore
from .strategy import ProductionFedProxStrategy
from infrastructure.shared.logging_setup import setup_logging as _setup_logging
from infrastructure.shared.health_server import HealthServer

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


def _load_standard_vocab() -> Dict:
    """Carrega standard_vocab.json como fallback quando não há checkpoint."""
    candidates = [
        os.getenv("FL_VOCAB_PATH"),
        str(CHECKPOINT_DIR / "standard_vocab.json"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                with open(path, encoding="utf-8") as f:
                    vocab = json.load(f)
                logger.info("standard_vocab_loaded path=%s size=%d", path, len(vocab))
                return vocab
            except Exception as exc:
                logger.warning("standard_vocab_load_error path=%s error=%s", path, exc)
    logger.warning(
        "no_standard_vocab — execute build_standard_vocab.py antes do treinamento; "
        "inferência será inoperante até o primeiro checkpoint com vocab"
    )
    return {}


def _save_checkpoint(path: Path, state: dict) -> None:
    """Salva checkpoint e grava SHA-256 para verificação de integridade."""
    import torch
    torch.save(state, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(sha256, encoding="utf-8")


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


def _make_server_components(context: Context) -> ServerAppComponents:
    """
    Factory chamada pelo SuperLink a cada execução de ServerApp.

    Lê run_config (pyproject.toml / --run-config), recupera estado da sessão
    anterior (se houver) e reconstrói a estratégia FedProx com tracker restaurado
    e pesos do último checkpoint carregados como initial_parameters.

    TLS é responsabilidade do flower-superlink — não é configurado aqui.
    """
    num_rounds = int(context.run_config.get("num-rounds", FED_CFG.num_rounds))
    min_clients = int(context.run_config.get("min-clients", FED_CFG.min_available_clients))
    proximal_mu = float(context.run_config.get("proximal-mu", FED_CFG.proximal_mu))
    local_epochs = int(context.run_config.get("local-epochs", FED_CFG.local_epochs))
    round_timeout = int(context.run_config.get("round-timeout-seconds", 300))

    # ── Recovery de estado ───────────────────────────────────────────────────
    state_path = LOG_DIR / "training_state.json"
    state_store = TrainingStateStore(state_path)
    previous_state = state_store.load()

    # Marca nova sessão como "running" imediatamente — se crashar, próximo load detecta
    previous_state.status = "running"
    state_store.save(previous_state)

    # ── Modelo: carrega checkpoint da sessão anterior se disponível ──────────
    model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
    initial_parameters: Optional[fl.common.Parameters] = None
    recovered_vocab: Dict = {}

    if previous_state.last_checkpoint:
        ckpt_path = Path(previous_state.last_checkpoint)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Novo formato: {"model_state": ..., "vocab": ...}
                # Legado: state_dict puro (sem chave "model_state")
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    state_dict      = ckpt["model_state"]
                    recovered_vocab = ckpt.get("vocab", {})
                else:
                    state_dict = ckpt
                    logger.warning(
                        "checkpoint_legacy_format — vocab ausente; "
                        "tentando carregar standard_vocab.json como fallback"
                    )
                model.load_state_dict(state_dict, strict=False)
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [v.cpu().detach().numpy().copy() for v in state_dict.values()]
                )
                logger.info(
                    "checkpoint_loaded_for_recovery",
                    extra={
                        "checkpoint": str(ckpt_path),
                        "last_round": previous_state.last_round,
                        "vocab_size": len(recovered_vocab),
                    },
                )
            except Exception as exc:
                logger.warning("checkpoint_load_error", extra={"error": str(exc)})

    # Fallback: se o checkpoint não trouxe vocab (primeiro round ou legado), carrega standard_vocab
    if not recovered_vocab:
        recovered_vocab = _load_standard_vocab()

    config_loader = get_config_loader()
    _health.start()

    strategy = ProductionFedProxStrategy(
        global_model=model,
        vocab=recovered_vocab,
        config_loader=config_loader,
        state_store=state_store,
        round_timeout=round_timeout,
        on_round_start=lambda rnd, cfg: write_health_status("running", round_num=rnd),
        on_round_complete=lambda rnd, metrics: _health.set_round_metrics(rnd, metrics),
        proximal_mu=proximal_mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
        fit_metrics_aggregation_fn=weighted_average_loss,
        on_fit_config_fn=lambda rnd: {
            "proximal_mu": proximal_mu,
            "local_epochs": local_epochs,
            "round": rnd,
        },
    )

    write_health_status("starting")
    logger.info(
        "server_startup_superlink",
        extra={
            "rounds": num_rounds,
            "min_clients": min_clients,
            "proximal_mu": proximal_mu,
            "round_timeout": round_timeout,
            "previous_status": previous_state.status,
            "recovered_from_round": previous_state.last_round,
        },
    )
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


# Entry point para: flwr run . <federation>
app = ServerApp(server_fn=_make_server_components)


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

    def _on_round_complete(self, round_num: int, metrics: dict) -> None:
        """Callback chamado pela strategy após aggregate_evaluate — publica no HealthServer."""
        _health.set_round_metrics(round_num, metrics)
        logger.debug("round_metrics_published", extra={"round": round_num})

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

        # Tenta recuperar vocab do checkpoint mais recente (se houver)
        recovered_vocab: Dict = {}
        existing_ckpts = sorted(CHECKPOINT_DIR.glob("round_*.pt"))
        if existing_ckpts:
            try:
                ckpt = torch.load(existing_ckpts[-1], map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    recovered_vocab = ckpt.get("vocab", {})
                    self.global_model.load_state_dict(ckpt["model_state"], strict=False)
                    logger.info(
                        "legacy_checkpoint_restored",
                        extra={"path": str(existing_ckpts[-1]), "vocab_size": len(recovered_vocab)},
                    )
            except Exception as exc:
                logger.warning("legacy_checkpoint_load_error", extra={"error": str(exc)})

        # Fallback: se não há checkpoint ou vocab ausente, carrega standard_vocab.json
        if not recovered_vocab:
            recovered_vocab = _load_standard_vocab()

        strategy = ProductionFedProxStrategy(
            global_model=self.global_model,
            vocab=recovered_vocab,
            config_loader=config_loader,
            on_round_start=self._on_round_start,
            on_round_complete=self._on_round_complete,
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

        from infrastructure.shared.tls import get_server_certs, tls_enabled
        certs = get_server_certs()
        logger.info(
            "server_tls",
            extra={"enabled": tls_enabled()},
        )

        try:
            fl.server.start_server(
                server_address=self.address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                certificates=certs,
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
    parser.add_argument(
        "--tls-cert-dir",
        default=None,
        help="Diretório com ca.crt, server.crt e server.key (omitir = sem TLS)",
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

    if args.tls_cert_dir:
        os.environ["FL_TLS_CERT_DIR"] = args.tls_cert_dir

    FederatedServer(
        address=address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        proximal_mu=args.mu,
    ).start()
