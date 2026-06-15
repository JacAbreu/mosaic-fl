"""
runner.py
Orquestrador do cliente Flower de produção (mosaicfl.v2).

Dois modos de execução:

  SuperNode (produção):
      flower-supernode --superlink <addr> --root-certificates ca.crt \
                       --node-config "client-id=hospital_1,data-source=sgbd"
    Expõe: app = ClientApp(client_fn=_client_fn)

  Legado (desenvolvimento local):
      python -m infrastructure.mosaicfl_client --client-id hospital_1
    Usa: ProductionClient + fl.client.start_client
"""
import argparse
import logging
import os
import random
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple

import flwr as fl
import torch
from flwr.client import ClientApp
from flwr.common import Context
from torch.utils.data import DataLoader, random_split

from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import FED_CFG, RUNTIME_CFG

from .datasource import DataSourceFactory
from .heartbeat import write_heartbeat
from infrastructure.shared.health_server import HealthServer

SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "localhost:8080")
CLIENT_ID = os.getenv("FL_CLIENT_ID", "client_0")
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
HEALTH_PORT = int(os.getenv("FL_HEALTH_PORT", "8081"))

HEARTBEAT_INTERVAL = int(os.getenv("FL_HEARTBEAT_INTERVAL", "60"))
RECONNECT_DELAY = int(os.getenv("FL_RECONNECT_DELAY", "30"))

logger = logging.getLogger(__name__)

_health = HealthServer(port=HEALTH_PORT)


def setup_logging(client_id: str) -> None:
    """Configura logging em arquivo rotativo e stdout (idempotente se já configurado)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if logging.getLogger().handlers:
        return
    file_handler = RotatingFileHandler(
        LOG_DIR / f"client_{client_id}.log",
        maxBytes=20 * 1024 * 1024,  # 20 MB por arquivo
        backupCount=5,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler(sys.stdout)],
    )


def parse_client_id(client_id: str) -> int:
    """Converte ID do hospital (string) para inteiro usado pelo FedProxClient."""
    try:
        return int(client_id)
    except ValueError:
        return abs(hash(client_id)) % 10_000


def _split_loader(loader: DataLoader, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Separa um DataLoader em treino e validação."""
    dataset = loader.dataset
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    if n_train < 1:
        return loader, loader
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(FED_CFG.random_seed),
    )
    return (
        DataLoader(train_ds, batch_size=loader.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=loader.batch_size, shuffle=False),
    )


# Cache de DataLoaders por tipo de fonte — evita recarregar o banco a cada round
_loader_cache: dict[str, Tuple[DataLoader, DataLoader]] = {}


def _client_fn(context: Context) -> fl.client.Client:
    """
    Factory chamada pelo SuperNode a cada round.

    Lê node_config (--node-config do flower-supernode) para identificar
    o hospital e a fonte de dados. TLS é responsabilidade do SuperNode — não
    é configurado aqui.

    DataLoaders são cacheados entre rounds: a query ao banco (que pode levar minutos)
    ocorre apenas no primeiro round. Dados não mudam durante uma sessão de treinamento.
    """
    client_id_str    = str(context.node_config.get("client-id", str(context.node_id)))
    data_source_type = str(
        context.node_config.get("data-source", os.getenv("FL_DATA_SOURCE", "simulated"))
    )
    client_id_int = parse_client_id(client_id_str)

    if data_source_type not in _loader_cache:
        logger.info(
            "data_loading_start",
            extra={"client_id": client_id_str, "data_source": data_source_type},
        )
        source = DataSourceFactory.create(data_source_type)
        _loader_cache[data_source_type] = _split_loader(source.load())
        logger.info(
            "data_loaded_and_cached",
            extra={"client_id": client_id_str, "data_source": data_source_type},
        )
    else:
        logger.debug("data_cache_hit source=%s", data_source_type)

    train_loader, val_loader = _loader_cache[data_source_type]

    return FedProxClient(
        client_id=client_id_int,
        train_loader=train_loader,
        val_loader=val_loader,
    ).to_client()


# Entry point para: flower-supernode ... (SuperNode executa flwr-clientapp internamente)
app = ClientApp(client_fn=_client_fn)


class ProductionClient:
    """Cliente Flower que roda continuamente no hospital."""

    def __init__(
        self,
        server_address: str = SERVER_ADDRESS,
        client_id: str = CLIENT_ID,
        data_source: Optional[str] = None,
    ):
        self.server_address = server_address
        self.client_id = client_id
        self.data_source = data_source
        self.client_id_int = parse_client_id(client_id)
        os.environ["FL_CLIENT_ID"] = client_id
        self.device = RUNTIME_CFG.device

    def _load_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Carrega treino/validação conforme FL_DATA_SOURCE ou --data-source."""
        source_type = self.data_source or os.getenv("FL_DATA_SOURCE", "simulated")
        logger.info("Carregando dados locais (fonte=%s)...", source_type)

        source = DataSourceFactory.create(source_type)

        meta = source.get_metadata()
        logger.info("Metadata da fonte: %s", meta)

        full_loader = source.load()
        return _split_loader(full_loader)

    def _build_flower_client(self) -> fl.client.NumPyClient:
        """Constrói FedProxClient v2 (modelo criado internamente)."""
        train_loader, val_loader = self._load_data_loaders()
        return FedProxClient(
            client_id=self.client_id_int,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    def _heartbeat_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            write_heartbeat("ready")
            time.sleep(HEARTBEAT_INTERVAL)

    def run(self) -> None:
        _health.start()
        _health.set_status("starting", client_id=self.client_id, server=self.server_address)

        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(stop_event,),
            daemon=True,
        )
        heartbeat_thread.start()

        data_source = self.data_source or os.getenv("FL_DATA_SOURCE", "simulated")
        logger.info("=" * 60)
        logger.info("MOSAIC-FL — CLIENTE DE PRODUÇÃO (v2) [%s]", self.client_id)
        logger.info("=" * 60)
        logger.info("Servidor:       %s", self.server_address)
        logger.info("Device:         %s", self.device)
        logger.info("Fonte de dados: %s", data_source)
        logger.info("Heartbeat:      a cada %ss", HEARTBEAT_INTERVAL)
        logger.info("=" * 60)

        _MAX_BACKOFF  = 1800  # 30 min
        _backoff_base = RECONNECT_DELAY
        _attempt      = 0

        while not stop_event.is_set():
            try:
                flower_client = self._build_flower_client()
                logger.info("Conectando ao servidor %s...", self.server_address)
                _health.set_status("connecting", client_id=self.client_id, server=self.server_address)
                from infrastructure.shared.tls import get_client_root_cert, tls_enabled
                root_cert = get_client_root_cert()
                logger.info("client_tls", extra={"enabled": tls_enabled()})
                fl.client.start_client(
                    server_address=self.server_address,
                    client=flower_client,
                    root_certificates=root_cert,
                )
                # Sessão concluída normalmente — reseta backoff
                _attempt = 0
                delay = _backoff_base + random.uniform(0, 5)
                _health.set_status("reconnecting", client_id=self.client_id, server=self.server_address)
                logger.info("Sessão concluída. Reconectando em %.0fs...", delay)
                time.sleep(delay)
            except KeyboardInterrupt:
                logger.info("Cliente interrompido pelo usuário.")
                stop_event.set()
                break
            except Exception as e:
                _attempt += 1
                delay = min(_backoff_base * (2 ** (_attempt - 1)), _MAX_BACKOFF)
                delay += random.uniform(0, min(delay * 0.1, 30))
                _health.set_status("error", client_id=self.client_id, error=str(e))
                logger.error("Erro na conexão (tentativa %d): %s", _attempt, e)
                logger.info("Backoff exponencial: reconectando em %.0fs...", delay)
                time.sleep(delay)

        heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL + 5)
        write_heartbeat("offline")
        logger.info("Cliente finalizado.")


def main() -> None:
    """CLI do cliente de produção (mosaicfl.v2)."""
    parser = argparse.ArgumentParser(
        description="Cliente Flower de produção — MOSAIC-FL (v2)",
    )
    parser.add_argument("--server", default=SERVER_ADDRESS, help="Endereço do servidor")
    parser.add_argument("--client-id", default=CLIENT_ID, help="ID único deste hospital")
    parser.add_argument(
        "--data-source",
        default=None,
        choices=["simulated", "sgbd", "csv"],
        help="Fonte de dados (default: FL_DATA_SOURCE ou simulated)",
    )
    parser.add_argument("--device", default=str(RUNTIME_CFG.device), help="Device PyTorch (informativo)")
    parser.add_argument(
        "--tls-cert-dir",
        default=None,
        help="Diretório com ca.crt (omitir = sem TLS)",
    )
    args = parser.parse_args()

    setup_logging(args.client_id)
    os.environ["FL_CLIENT_ID"] = args.client_id
    if args.data_source:
        os.environ["FL_DATA_SOURCE"] = args.data_source
    if args.tls_cert_dir:
        os.environ["FL_TLS_CERT_DIR"] = args.tls_cert_dir

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["FL_LOG_DIR"] = str(LOG_DIR)

    ProductionClient(
        server_address=args.server,
        client_id=args.client_id,
        data_source=args.data_source,
    ).run()


if __name__ == "__main__":
    main()
