import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.data_loader import DataLoadError


class TestDataLoadError:

    def test_includes_attempts_in_message(self):
        attempts = [{"fonte": "SGBD", "erro": "timeout"}, {"fonte": "CSV", "erro": "not found"}]
        err = DataLoadError("Falha total", attempts=attempts)
        msg = str(err)
        assert "SGBD" in msg
        assert "CSV" in msg
        assert "timeout" in msg

    def test_without_attempts(self):
        err = DataLoadError("Falha simples")
        assert "Falha simples" in str(err)
