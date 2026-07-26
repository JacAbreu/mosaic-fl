"""
Testes para _setup_clientapp_logging() em infrastructure/mosaicfl_client/runner/supernode.py.

flwr-clientapp roda como subprocesso próprio a cada rodada, sem nenhum handler de
logging configurado — sem isso, logger.info() cai no logging.lastResort do Python
(só WARNING+ sobrevive) e toda a lógica de treino real (client_fit,
local_calibration_fit, client_resources) fica invisível em qualquer log capturado.
Achado 2026-07-25 — mesmo gap já corrigido do lado do servidor em 2026-07-05/06
(ver infrastructure/mosaicfl_server/runner/superlink.py).
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.mosaicfl_client.runner.supernode import _setup_clientapp_logging


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    """Isola cada teste: remove handlers _mosaicfl_clientapp adicionados por testes
    anteriores (o root logger é global/compartilhado entre módulos)."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for h in list(root.handlers):
        if getattr(h, "_mosaicfl_clientapp", False) and h not in before:
            root.removeHandler(h)
            h.close()


class TestSetupClientappLogging:
    def test_attaches_file_handler_to_root_logger(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_clientapp_logging("BPSP")
        root = logging.getLogger()
        handlers = [h for h in root.handlers if getattr(h, "_mosaicfl_clientapp", False)]
        assert len(handlers) == 1

    def test_creates_log_file_under_experiments_logs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_clientapp_logging("HSL")
        log_files = list((tmp_path / "experiments" / "logs").glob("clientapp_HSL_*.log"))
        assert len(log_files) == 1

    def test_info_log_reaches_the_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_clientapp_logging("BPSP")
        logging.getLogger("mosaicfl.core.client").info("client_fit client_id=0 loss=0.5")
        log_files = list((tmp_path / "experiments" / "logs").glob("clientapp_BPSP_*.log"))
        content = log_files[0].read_text(encoding="utf-8")
        assert "client_fit client_id=0 loss=0.5" in content

    def test_idempotent_does_not_duplicate_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_clientapp_logging("BPSP")
        _setup_clientapp_logging("BPSP")
        root = logging.getLogger()
        handlers = [h for h in root.handlers if getattr(h, "_mosaicfl_clientapp", False)]
        assert len(handlers) == 1

    def test_root_logger_level_set_to_info(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_clientapp_logging("BPSP")
        assert logging.getLogger().level == logging.INFO
