import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.data_loader import _map_columns


class TestMapColumns:

    def test_renames_known_columns(self):
        df = pd.DataFrame({"hospital": ["A"], "evolucao": [0], "sintomas": ["febre"]})
        mapped = _map_columns(df)
        assert "instituicao" in mapped.columns
        assert "desfecho" in mapped.columns

    def test_case_insensitive(self):
        df = pd.DataFrame({"HOSPITAL": ["A"], "EVOLUCAO": [0]})
        mapped = _map_columns(df)
        assert "instituicao" in mapped.columns

    def test_preserves_unmapped_columns(self):
        df = pd.DataFrame({"hospital": ["A"], "coluna_extra": [1]})
        mapped = _map_columns(df)
        assert "coluna_extra" in mapped.columns
