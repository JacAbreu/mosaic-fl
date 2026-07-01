"""
runner — Orquestrador do cliente Flower de produção (mosaicfl.v2).

Dois modos de execução:

  SuperNode (produção):
      flower-supernode --superlink <addr> --root-certificates ca.crt \
                       --node-config "client-id=hospital_1,data-source=sgbd"
    Expõe: app = ClientApp(client_fn=_client_fn)

  Legado (desenvolvimento local):
      python -m infrastructure.mosaicfl_client --client-id hospital_1
    Usa: ProductionClient + fl.client.start_client

Submódulos:
  config.py         — constantes de ambiente, singleton HealthServer, setup_logging
  data_utils.py        — parse_client_id, _split_loader, cache de DataLoaders
  supernode.py            — _client_fn + app (ClientApp)
  legacy_client.py           — ProductionClient
  cli.py                        — main() (CLI do modo legado)
"""
from ..datasource import DataSourceFactory
from .cli import main
from .config import CLIENT_ID, HEALTH_PORT, LOG_DIR, SERVER_ADDRESS, setup_logging
from .data_utils import parse_client_id
from .legacy_client import ProductionClient
from .supernode import _client_fn, app

__all__ = [
    "app",
    "ProductionClient",
    "main",
    "setup_logging",
    "parse_client_id",
    "DataSourceFactory",
    "_client_fn",
    "SERVER_ADDRESS",
    "CLIENT_ID",
    "LOG_DIR",
    "HEALTH_PORT",
]
