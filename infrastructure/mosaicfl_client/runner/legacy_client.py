"""legacy_client.py — Cliente Flower legado para desenvolvimento local (python -m infrastructure.mosaicfl_client)."""
import logging
import os
import random
import threading
import time
from typing import Optional, Tuple

import flwr as fl
from torch.utils.data import DataLoader

from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import RUNTIME_CFG

from ..datasource import DataSourceFactory
from ..heartbeat import write_heartbeat
from .config import CLIENT_ID, HEARTBEAT_INTERVAL, RECONNECT_DELAY, SERVER_ADDRESS, _health
from .data_utils import _split_loader, parse_client_id

logger = logging.getLogger(__name__)


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
        logger.info("Carregando dados locais (fonte=%s, hospital=%s)...", source_type, self.client_id)

        source = DataSourceFactory.create(source_type, hospital_id=self.client_id)

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
