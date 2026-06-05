"""
client_availability_checker.py
Verifica quais clientes (hospitais) estão online e prontos para treinar.
"""
import json
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ClientAvailabilityChecker:
    """
    Verifica quais clientes (hospitais) estão online e prontos para treinar.

    Estratégias:
      1. Registry file: lê arquivo de heartbeat escrito pelos clientes
      2. TCP ping: tenta conectar em cada cliente conhecido
    """

    def __init__(self, known_clients: Optional[List[str]] = None):
        # Lista de clientes conhecidos (IPs/hosts dos hospitais)
        self.known_clients = known_clients or []
        self.client_registry: Dict[str, dict] = {}  # {client_id: {last_seen, status}}

    def check_via_server(self, registry_path: str = "logs/client_registry.json") -> Tuple[int, List[str]]:
        """
        Lê arquivo de registro compartilhado para ver clientes ativos.

        Em produção real, substituir por:
          - Query REST/gRPC ao servidor Flower
          - Prometheus metrics
          - Consul/etcd service discovery

        Retorna: (num_clients, [client_ids])
        """
        registry_file = Path(registry_path)
        if not registry_file.exists():
            logger.debug("Registry de clientes não encontrado. Nenhum cliente ativo.")
            return 0, []

        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)

            # Filtra clientes que reportaram nos últimos 10 minutos
            now = datetime.now().timestamp()
            active = [
                cid for cid, info in registry.items()
                if isinstance(info, dict) and (now - info.get("last_seen", 0)) < 600  # 10 min
            ]
            logger.info("clients_active", extra={"count": len(active), "clients": active})
            return len(active), active
        except Exception as e:
            logger.warning("registry_read_error", extra={"error": str(e)})
            return 0, []

    def check_via_ping(self, timeout: float = 5.0) -> Tuple[int, List[str]]:
        """Faz ping TCP em cada cliente conhecido."""
        active = []
        for client_addr in self.known_clients:
            try:
                host, port = (
                    client_addr.split(":") if ":" in client_addr else (client_addr, "8081")
                )
                sock = socket.create_connection((host, int(port)), timeout=timeout)
                sock.close()
                active.append(client_addr)
                logger.debug("client_ping_ok", extra={"client": client_addr})
            except Exception:
                logger.debug("client_ping_fail", extra={"client": client_addr})

        return len(active), active

    def register_client(self, client_id: str, host: str, port: int = 8081):
        """Registra um novo cliente conhecido (evita duplicatas)."""
        addr = f"{host}:{port}"
        if addr not in self.known_clients:
            self.known_clients.append(addr)
            logger.info("client_registered", extra={"client_id": client_id, "address": addr})
        else:
            logger.debug("client_already_registered", extra={"client_id": client_id, "address": addr})