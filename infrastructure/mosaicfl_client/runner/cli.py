"""cli.py — Entry point de linha de comando do cliente legado (python -m infrastructure.mosaicfl_client)."""
import argparse
import os

from mosaicfl.core.config import RUNTIME_CFG

from .config import CLIENT_ID, LOG_DIR, SERVER_ADDRESS, setup_logging
from .legacy_client import ProductionClient


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
