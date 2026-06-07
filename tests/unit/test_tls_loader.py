import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.shared.tls import (
    load_client_root_cert,
    load_server_certs,
    get_server_certs,
    get_client_root_cert,
    tls_enabled,
)


def _write_fake_certs(directory: Path) -> None:
    """Cria arquivos de certificado fictícios para testes."""
    (directory / "ca.crt").write_bytes(b"fake-ca-cert")
    (directory / "server.crt").write_bytes(b"fake-server-cert")
    (directory / "server.key").write_bytes(b"fake-server-key")


class TestLoadServerCerts:

    def test_returns_three_byte_strings(self, tmp_path):
        _write_fake_certs(tmp_path)
        ca, cert, key = load_server_certs(tmp_path)
        assert ca == b"fake-ca-cert"
        assert cert == b"fake-server-cert"
        assert key == b"fake-server-key"

    def test_raises_if_ca_missing(self, tmp_path):
        (tmp_path / "server.crt").write_bytes(b"x")
        (tmp_path / "server.key").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="ca.crt"):
            load_server_certs(tmp_path)

    def test_raises_if_server_cert_missing(self, tmp_path):
        (tmp_path / "ca.crt").write_bytes(b"x")
        (tmp_path / "server.key").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="server.crt"):
            load_server_certs(tmp_path)

    def test_raises_if_server_key_missing(self, tmp_path):
        (tmp_path / "ca.crt").write_bytes(b"x")
        (tmp_path / "server.crt").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="server.key"):
            load_server_certs(tmp_path)


class TestLoadClientRootCert:

    def test_returns_ca_bytes(self, tmp_path):
        _write_fake_certs(tmp_path)
        ca = load_client_root_cert(tmp_path)
        assert ca == b"fake-ca-cert"

    def test_raises_if_ca_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ca.crt"):
            load_client_root_cert(tmp_path)


class TestGetServerCerts:

    def test_raises_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("FL_TLS_CERT_DIR", raising=False)
        with pytest.raises(EnvironmentError, match="FL_TLS_CERT_DIR"):
            get_server_certs()

    def test_returns_tuple_when_env_set(self, tmp_path, monkeypatch):
        _write_fake_certs(tmp_path)
        monkeypatch.setenv("FL_TLS_CERT_DIR", str(tmp_path))
        result = get_server_certs()
        assert result is not None
        ca, cert, key = result
        assert isinstance(ca, bytes)
        assert isinstance(cert, bytes)
        assert isinstance(key, bytes)

    def test_raises_file_not_found_when_dir_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FL_TLS_CERT_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            get_server_certs()


class TestGetClientRootCert:

    def test_raises_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("FL_TLS_CERT_DIR", raising=False)
        with pytest.raises(EnvironmentError, match="FL_TLS_CERT_DIR"):
            get_client_root_cert()

    def test_returns_bytes_when_env_set(self, tmp_path, monkeypatch):
        _write_fake_certs(tmp_path)
        monkeypatch.setenv("FL_TLS_CERT_DIR", str(tmp_path))
        result = get_client_root_cert()
        assert result == b"fake-ca-cert"

    def test_raises_if_ca_missing_in_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FL_TLS_CERT_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            get_client_root_cert()


class TestTlsEnabled:

    def test_false_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("FL_TLS_CERT_DIR", raising=False)
        assert tls_enabled() is False

    def test_true_when_env_set(self, monkeypatch):
        monkeypatch.setenv("FL_TLS_CERT_DIR", "/some/path")
        assert tls_enabled() is True
