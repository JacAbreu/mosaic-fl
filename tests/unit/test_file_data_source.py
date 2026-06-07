import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.v2.data_loader import FileDataSource


class TestFileDataSource:

    def test_not_available_when_empty(self, tmp_path):
        src = FileDataSource(base_dir=tmp_path, filenames=["nao_existe.csv"])
        assert not src.is_available()

    def test_available_when_file_exists(self, tmp_path):
        f = tmp_path / "dataset.csv"
        pd.DataFrame({"a": [1]}).to_csv(f, index=False)
        src = FileDataSource(base_dir=tmp_path, filenames=["dataset.csv"])
        assert src.is_available()

    def test_loads_csv(self, tmp_path):
        f = tmp_path / "dataset.csv"
        expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        expected.to_csv(f, index=False)
        src = FileDataSource(base_dir=tmp_path, filenames=["dataset.csv"])
        df = src.load()
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_raises_when_not_found(self, tmp_path):
        src = FileDataSource(base_dir=tmp_path, filenames=["nao_existe.csv"])
        with pytest.raises(FileNotFoundError):
            src.load()
