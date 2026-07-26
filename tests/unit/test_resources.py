"""
Testes para mosaicfl.core.resources.sample_gpu_power_w() — amostragem best-effort
de potência da GPU via nvidia-smi, compartilhada entre Caminho A (manual_loop.py)
e Caminho B (client.py).
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.resources import sample_gpu_power_w


class TestSampleGpuPowerW:
    @patch("mosaicfl.core.resources.subprocess.run")
    def test_returns_power_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="123.45\n")
        assert sample_gpu_power_w() == 123.45

    @patch("mosaicfl.core.resources.subprocess.run")
    def test_returns_none_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert sample_gpu_power_w() is None

    @patch("mosaicfl.core.resources.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_nvidia_smi_absent(self, mock_run):
        assert sample_gpu_power_w() is None

    @patch("mosaicfl.core.resources.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.0))
    def test_returns_none_on_timeout(self, mock_run):
        assert sample_gpu_power_w() is None

    @patch("mosaicfl.core.resources.subprocess.run")
    def test_returns_none_on_malformed_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not-a-number\n")
        assert sample_gpu_power_w() is None

    @patch("mosaicfl.core.resources.subprocess.run")
    def test_returns_none_on_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        assert sample_gpu_power_w() is None
