"""
tls.py
Utilitários de TLS para comunicação segura via gRPC (Flower).

TLS é obrigatório. FL_TLS_CERT_DIR deve estar definido antes de iniciar
qualquer servidor ou cliente. Ausência da variável lança EnvironmentError.

Uso:
    # Servidor
    from infrastructure.shared.tls import get_server_certs
    certs = get_server_certs()   # raises EnvironmentError se FL_TLS_CERT_DIR ausente
    fl.server.start_server(..., certificates=certs)

    # Cliente
    from infrastructure.shared.tls import get_client_root_cert
    root_cert = get_client_root_cert()   # raises EnvironmentError se FL_TLS_CERT_DIR ausente
    fl.client.start_client(..., root_certificates=root_cert)

Variáveis de ambiente:
    FL_TLS_CERT_DIR  — diretório com os certificados (obrigatório em produção)

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


def get_server_certs() -> tuple[bytes, bytes, bytes]:
    """Lê FL_TLS_CERT_DIR e carrega certificados do servidor.

    Raises:
        EnvironmentError: se FL_TLS_CERT_DIR não estiver definido.
        FileNotFoundError: se algum arquivo de certificado estiver ausente.
    """
    cert_dir_str = os.getenv(_ENV_KEY)
    if not cert_dir_str:
        raise EnvironmentError(
            "FL_TLS_CERT_DIR não definido. "
            "TLS é obrigatório. Configure o diretório de certificados ou execute "
            "scripts/gen_certs.sh para gerar certificados de desenvolvimento."
        )
    return load_server_certs(Path(cert_dir_str))


def get_client_root_cert() -> bytes:
    """Lê FL_TLS_CERT_DIR e carrega o certificado raiz para o cliente.

    Raises:
        EnvironmentError: se FL_TLS_CERT_DIR não estiver definido.
        FileNotFoundError: se ca.crt estiver ausente.
    """
    cert_dir_str = os.getenv(_ENV_KEY)
    if not cert_dir_str:
        raise EnvironmentError(
            "FL_TLS_CERT_DIR não definido. "
            "TLS é obrigatório. Configure o diretório de certificados ou execute "
            "scripts/gen_certs.sh para gerar certificados de desenvolvimento."
        )
    return load_client_root_cert(Path(cert_dir_str))


def tls_enabled() -> bool:
    """Retorna True se FL_TLS_CERT_DIR está configurado."""
    return bool(os.getenv(_ENV_KEY))
