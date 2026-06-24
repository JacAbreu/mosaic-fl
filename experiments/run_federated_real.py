"""
run_federated_real.py — Rede federada real: desktop (servidor + BPSP) + notebook (cliente HSL).

Diferença em relação a run_experiments_simulation.py
------------------------------------------------------
A simulação roda servidor e todos os clientes no mesmo processo, na mesma máquina,
usando fl.simulation.start_simulation(). Os dados ficam no mesmo banco. Nada viaja
pela rede. É academicamente válido, mas não demonstra privacidade de dados.

Este script usa fl.server.start_server() e fl.client.start_client() — sockets TCP reais.
Os pesos do modelo trafegam pela rede a cada round (~2.8 MB × 2 direções × N rounds).
Os dados clínicos NUNCA saem de cada máquina. Isso é FL de verdade.

Setup da rede local
-------------------
  Desktop  IP: descobrir com `ip addr show` ou `hostname -I`
  Notebook IP: qualquer — só precisa enxergar o desktop
  Porta:    8080 (configurável via FL_SERVER_PORT)
  Ambos devem estar na mesma rede Wi-Fi ou cabo.

Como usar
---------
  Desktop  → python experiments/run_federated_real.py --mode server
  Notebook → python experiments/run_federated_real.py --mode client --server 192.168.X.X:8080

  Para verificar conectividade antes de iniciar:
  Notebook → python experiments/run_federated_real.py --check --server 192.168.X.X:8080

Variáveis de ambiente relevantes
---------------------------------
  FL_DB_URL          URL PostgreSQL com os dados do hospital local
  FL_HOSPITAL_ID     Identificador deste nó (HSL | BPSP | etc.)
  FL_SERVER_PORT     Porta do servidor (padrão: 8080)
  FL_NUM_ROUNDS      Número de rounds (padrão: config)
  FL_MIN_CLIENTS     Mínimo de clientes para iniciar cada round (padrão: 2)
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Verificação de conectividade
# ---------------------------------------------------------------------------

def _check_connectivity(server: str) -> None:
    """Verifica se o notebook consegue enxergar o servidor antes de iniciar."""
    host, _, port_str = server.rpartition(":")
    port = int(port_str) if port_str.isdigit() else 8080
    host = host or "localhost"

    print(f"\nVerificando conectividade com {host}:{port}...")
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"  OK — servidor alcançável em {host}:{port}")
            print("  Pode iniciar o cliente.\n")
    except ConnectionRefusedError:
        print(f"  RECUSADO — {host}:{port} está acessível mas nenhum servidor está escutando.")
        print("  Inicie o servidor no desktop primeiro:\n")
        print(f"    make fl-server\n")
        sys.exit(1)
    except OSError as e:
        print(f"  INACESSÍVEL — não foi possível alcançar {host}:{port}: {e}")
        print("\n  Verifique:")
        print("  1. Desktop e notebook estão na mesma rede Wi-Fi?")
        print(f"  2. Firewall do desktop permite a porta {port}?")
        print(f"       Ubuntu/Debian:  sudo ufw allow {port}/tcp")
        print(f"       Fedora/RHEL:    sudo firewall-cmd --add-port={port}/tcp --permanent")
        print(f"  3. IP do desktop está correto? (descobrir: hostname -I)")
        print(f"  4. O servidor foi iniciado? (make fl-server no desktop)\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Modo servidor (desktop)
# ---------------------------------------------------------------------------

def _run_server(args: argparse.Namespace) -> None:
    port    = int(os.getenv("FL_SERVER_PORT", str(args.port)))
    rounds  = int(os.getenv("FL_NUM_ROUNDS",  str(args.rounds)))
    min_cli = int(os.getenv("FL_MIN_CLIENTS", str(args.min_clients)))

    hospital_id = os.getenv("FL_HOSPITAL_ID", args.hospital_id or "BPSP")
    db_url      = os.getenv("FL_DB_URL", "")

    print("=" * 60)
    print("  MOSAIC-FL — SERVIDOR (desktop)")
    print("=" * 60)
    print(f"  Endereço:       0.0.0.0:{port}")
    print(f"  Rounds:         {rounds}")
    print(f"  Mín. clientes:  {min_cli}")
    print(f"  Hospital local: {hospital_id}")
    print(f"  DB:             {'configurado' if db_url else 'NÃO configurado — dados sintéticos'}")
    print()

    if not db_url:
        print("  AVISO: FL_DB_URL não configurado. O servidor usará dados sintéticos.")
        print("  Configure FL_DB_URL para usar os dados reais do FAPESP.\n")

    _print_ip_hint(port)

    # Importa e inicia o servidor de produção
    from infrastructure.mosaicfl_server.runner import FederatedServer
    FederatedServer(
        address=f"0.0.0.0:{port}",
        num_rounds=rounds,
        min_clients=min_cli,
    ).start()


# ---------------------------------------------------------------------------
# Modo cliente (notebook)
# ---------------------------------------------------------------------------

def _run_client(args: argparse.Namespace) -> None:
    server      = args.server
    hospital_id = os.getenv("FL_HOSPITAL_ID", args.hospital_id or "HSL")
    db_url      = os.getenv("FL_DB_URL", "")
    data_source = "sgbd" if db_url else "simulated"

    print("=" * 60)
    print("  MOSAIC-FL — CLIENTE (notebook)")
    print("=" * 60)
    print(f"  Servidor:       {server}")
    print(f"  Hospital local: {hospital_id}")
    print(f"  Fonte de dados: {data_source}")
    print(f"  DB:             {'configurado' if db_url else 'NÃO configurado — dados sintéticos'}")
    print()

    if not db_url:
        print("  AVISO: FL_DB_URL não configurado. O cliente usará dados sintéticos.")
        print("  Configure FL_DB_URL=postgresql://... para usar os dados reais do FAPESP.\n")

    # Verifica conectividade antes de tentar conectar
    _check_connectivity(server)

    from infrastructure.mosaicfl_client.runner import ProductionClient
    ProductionClient(
        server_address=server,
        client_id=hospital_id,
        data_source=data_source,
    ).run()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _print_ip_hint(port: int) -> None:
    """Imprime o IP local para facilitar a configuração do notebook."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        print(f"  IP deste desktop: {local_ip}")
        print(f"  No notebook, execute:")
        print(f"    make fl-client FL_SERVER={local_ip}:{port} FL_HOSPITAL_ID=HSL\n")
    except Exception:
        print("  (não foi possível detectar o IP local automaticamente)")
        print("  Execute `hostname -I` para descobrir o IP deste desktop.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOSAIC-FL — Rede federada real (desktop + notebook)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Desktop — inicia servidor aguardando clientes
  python experiments/run_federated_real.py --mode server

  # Notebook — conecta ao servidor do desktop
  python experiments/run_federated_real.py --mode client --server 192.168.1.100:8080

  # Notebook — verifica conectividade antes de iniciar
  python experiments/run_federated_real.py --check --server 192.168.1.100:8080
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["server", "client"],
        help="'server' no desktop, 'client' no notebook",
    )
    parser.add_argument(
        "--server",
        default=os.getenv("FL_SERVER", "localhost:8080"),
        metavar="HOST:PORT",
        help="Endereço do servidor (padrão: FL_SERVER ou localhost:8080)",
    )
    parser.add_argument(
        "--hospital-id",
        default=os.getenv("FL_HOSPITAL_ID"),
        help="ID deste hospital/nó (padrão: BPSP no server, HSL no client)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FL_SERVER_PORT", "8080")),
        help="Porta do servidor (apenas no modo server, padrão: 8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.getenv("FL_NUM_ROUNDS", "20")),
        help="Número de rounds FL (apenas no modo server)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=int(os.getenv("FL_MIN_CLIENTS", "2")),
        help="Mínimo de clientes para iniciar cada round (padrão: 2)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Apenas verifica conectividade com o servidor, não inicia o cliente",
    )

    args = parser.parse_args()

    if args.check:
        _check_connectivity(args.server)
        return

    if not args.mode:
        parser.error("Informe --mode server (desktop) ou --mode client (notebook)")

    if args.mode == "server":
        _run_server(args)
    else:
        _run_client(args)


if __name__ == "__main__":
    main()
