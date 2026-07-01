"""
runner — Orquestrador do servidor Flower de produção (mosaicfl.v2).

Dois modos de execução:

  SuperLink (produção):
      flower-superlink ...         # infraestrutura persistente
      flwr run . local             # dispara ServerApp via SuperLink
    Expõe: app = ServerApp(server_fn=_make_server_components)

  Legado (desenvolvimento local):
      python -m infrastructure.mosaicfl_server --port 8080
    Usa: FederatedServer + fl.server.start_server

Submódulos:
  config.py           — constantes de ambiente, singleton HealthServer, setup_logging
  checkpoint_io.py       — _load_standard_vocab, _save_checkpoint
  health.py                — write_health_status
  superlink.py                — _make_server_components + app (ServerApp)
  legacy_server.py               — FederatedServer
  cli.py                            — main() (CLI do modo legado)

Nota: `app` é referenciado por caminho de string em pyproject.toml
("infrastructure.mosaicfl_server.runner:app") — deve permanecer re-exportado aqui.
`_save_checkpoint` é importado localmente por strategy.py (import tardio, evita
ciclo de import já que runner → strategy no carregamento de superlink.py).
"""
from .checkpoint_io import _load_standard_vocab, _save_checkpoint
from .cli import main
from .config import CHECKPOINT_DIR, HEALTH_PORT, LOG_DIR, SERVER_ADDRESS, setup_logging
from .health import write_health_status
from .legacy_server import FederatedServer
from .superlink import _make_server_components, app

__all__ = [
    "app",
    "FederatedServer",
    "main",
    "setup_logging",
    "write_health_status",
    "_load_standard_vocab",
    "_save_checkpoint",
    "_make_server_components",
    "SERVER_ADDRESS",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "HEALTH_PORT",
]
