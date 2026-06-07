import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.data_loader import _generate_synthetic_fallback


class TestGenerateSyntheticFallback:

    def test_returns_correct_size(self):
        df = _generate_synthetic_fallback(200)
        assert len(df) == 200

    def test_has_required_columns(self):
        df = _generate_synthetic_fallback(50)
        for col in ["instituicao", "idade", "sintoma", "exame", "desfecho"]:
            assert col in df.columns

    def test_desfecho_binary(self):
        df = _generate_synthetic_fallback(100)
        assert set(df["desfecho"].unique()).issubset({0, 1})

    def test_is_reproducible(self):
        df1 = _generate_synthetic_fallback(100)
        df2 = _generate_synthetic_fallback(100)
        assert (df1["desfecho"].values == df2["desfecho"].values).all()
