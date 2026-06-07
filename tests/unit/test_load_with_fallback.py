import sys
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.data_loader import DataLoadError, load_with_fallback


class TestLoadWithFallback:

    def test_uses_synthetic_when_no_source(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=50)
        assert len(df) == 50
        assert df["_fonte"].iloc[0] == "sintetico"

    def test_raises_when_no_synthetic_allowed(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            with pytest.raises(DataLoadError):
                load_with_fallback(allow_synthetic=False)

    def test_explicit_csv_not_found_raises(self, tmp_path):
        nonexistent = str(tmp_path / "nao_existe.csv")
        with pytest.raises(FileNotFoundError, match="CSV informado não encontrado"):
            load_with_fallback(csv_path=nonexistent)

    def test_explicit_csv_loads_when_found(self, tmp_path):
        f = tmp_path / "base.csv"
        df = pd.DataFrame({
            "instituicao": ["H1", "H2"],
            "desfecho": [0, 1],
            "sintoma": ["febre", "tosse"],
        })
        df.to_csv(f, index=False)
        result = load_with_fallback(csv_path=str(f))
        assert result["_fonte"].iloc[0] == "csv_explicito"
        assert len(result) == 2

    def test_returns_fonte_column(self):
        df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=20)
        assert "_fonte" in df.columns

    def test_sgbd_skipped_when_no_url(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=10)
        assert df["_fonte"].iloc[0] == "sintetico"

    def test_sgbd_fails_gracefully_on_bad_url(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": "postgresql://invalid:5432/db"}):
            with patch("mosaicfl.v2.data_loader.DatabaseDataSource.is_available",
                       return_value=False):
                df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=10)
        assert df["_fonte"].iloc[0] == "sintetico"
