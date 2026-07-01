"""cli.py — Entry point de linha de comando do servidor legado (python -m infrastructure.mosaicfl_server)."""
import argparse
import os
from pathlib import Path

from mosaicfl.core.config import FED_CFG

from .config import CHECKPOINT_DIR, LOG_DIR, SERVER_ADDRESS, setup_logging
from .legacy_server import FederatedServer


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
