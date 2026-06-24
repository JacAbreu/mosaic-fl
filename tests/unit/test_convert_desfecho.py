import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.data_loader import _convert_desfecho


class TestConvertDesfecho:

    def test_text_to_numeric(self):
        # 0=alta  1=internacao_prolongada  2=uti  3=obito
        df = pd.DataFrame({"desfecho": ["alta", "obito", "alta", "uti"]})
        result = _convert_desfecho(df)
        assert result["desfecho"].tolist() == [0, 3, 0, 2]

    def test_numeric_unchanged(self):
        df = pd.DataFrame({"desfecho": [0, 1, 2, 3]})
        result = _convert_desfecho(df)
        assert result["desfecho"].tolist() == [0, 1, 2, 3]

    def test_preserves_original_column(self):
        df = pd.DataFrame({"desfecho": ["alta", "obito"]})
        result = _convert_desfecho(df)
        assert "desfecho_original" in result.columns
