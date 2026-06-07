"""
tls.py
Utilitários de TLS para comunicação segura via gRPC (Flower).

Uso:
    # Servidor
    from infrastructure.tls import get_server_certs
    certs = get_server_certs()          # None se FL_TLS_CERT_DIR não definido
    fl.server.start_server(..., certificates=certs)

    # Cliente
    from infrastructure.tls import get_client_root_cert
    root_cert = get_client_root_cert()  # None se FL_TLS_CERT_DIR não definido
    fl.client.start_client(..., root_certificates=root_cert)

Variáveis de ambiente:
    FL_TLS_CERT_DIR  — diretório com os certificados (ver estrutura abaixo)
                       se ausente, a conexão é insegura (adequado para rede local)

Estrutura esperada do diretório:
    $FL_TLS_CERT_DIR/
        ca.crt          ← CA raiz (servidor + cliente precisam)
        server.crt      ← certificado do servidor (só o servidor precisa)
        server.key      ← chave privada do servidor (só o servidor precisa)

Gerar certificados de desenvolvimento:
    bash scripts/gen_certs.sh [output_dir]
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_KEY = "FL_TLS_CERT_DIR"

_CA_FILE = "ca.crt"
_SERVER_CERT_FILE = "server.crt"
_SERVER_KEY_FILE = "server.key"


def _read(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Certificado TLS não encontrado: {path}")
    return path.read_bytes()


def load_server_certs(cert_dir: Path) -> tuple[bytes, bytes, bytes]:
    """Carrega os três arquivos necessários para o servidor gRPC com TLS.

    Returns:
        (ca_cert, server_cert, server_key) como bytes.

    Raises:
        FileNotFoundError: se algum arquivo estiver ausente.
    """
    ca = _read(cert_dir / _CA_FILE)
    cert = _read(cert_dir / _SERVER_CERT_FILE)
    key = _read(cert_dir / _SERVER_KEY_FILE)
    logger.info(
        "tls_server_certs_loaded",
        extra={"cert_dir": str(cert_dir), "ca_size": len(ca), "cert_size": len(cert)},
    )
    return ca, cert, key


def load_client_root_cert(cert_dir: Path) -> bytes:
    """Carrega o certificado da CA para o cliente verificar o servidor.

    Returns:
        Bytes do ca.crt.

    Raises:
        FileNotFoundError: se ca.crt estiver ausente.
    """
    ca = _read(cert_dir / _CA_FILE)
    logger.info(
        "tls_client_cert_loaded",
        extra={"cert_dir": str(cert_dir), "ca_size": len(ca)},
    )
    return ca


def get_server_certs() -> Optional[tuple[bytes, bytes, bytes]]:
    """Lê FL_TLS_CERT_DIR e carrega certificados do servidor.

    Returns None se FL_TLS_CERT_DIR não estiver definido (modo inseguro).
    Lança FileNotFoundError se a variável está definida mas os arquivos faltam.
    """
    cert_dir_str = os.getenv(_ENV_KEY)
    if not cert_dir_str:
        logger.warning("tls_disabled: FL_TLS_CERT_DIR não definido — canal gRPC inseguro")
        return None
    return load_server_certs(Path(cert_dir_str))


def get_client_root_cert() -> Optional[bytes]:
    """Lê FL_TLS_CERT_DIR e carrega o certificado raiz para o cliente.

    Returns None se FL_TLS_CERT_DIR não estiver definido (modo inseguro).
    """
    cert_dir_str = os.getenv(_ENV_KEY)
    if not cert_dir_str:
        logger.warning("tls_disabled: FL_TLS_CERT_DIR não definido — canal gRPC inseguro")
        return None
    return load_client_root_cert(Path(cert_dir_str))


def tls_enabled() -> bool:
    """Retorna True se FL_TLS_CERT_DIR está configurado."""
    return bool(os.getenv(_ENV_KEY))
