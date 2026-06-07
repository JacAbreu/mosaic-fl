import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from infrastructure.mosaicfl_server.config_loader import FileConfigLoader


class TestFileConfigLoader:

    # ── load ──────────────────────────────────────────────────────────────────

    def test_load_missing_file_returns_empty_dict(self, tmp_path):
        loader = FileConfigLoader(path=tmp_path / "nao_existe.json")
        assert loader.load(round_num=1) == {}

    def test_load_valid_json_returns_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"proximal_mu": 0.005, "stop": False}))
        loader = FileConfigLoader(path=config_file)
        result = loader.load(round_num=1)
        assert result["proximal_mu"] == pytest.approx(0.005)
        assert result["stop"] is False

    def test_load_corrupted_json_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text("{ chave: sem aspas }")
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == {}

    def test_load_empty_file_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text("")
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == {}

    def test_load_roundtrip_preserves_types(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        original = {"proximal_mu": 0.01, "pause_seconds": 30.0, "stop": True}
        config_file.write_text(json.dumps(original))
        loader = FileConfigLoader(path=config_file)
        result = loader.load(round_num=2)
        assert result == original

    def test_load_result_independent_of_round_num(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"stop": False}))
        loader = FileConfigLoader(path=config_file)
        assert loader.load(round_num=1) == loader.load(round_num=99)

    # ── write ─────────────────────────────────────────────────────────────────

    def test_write_creates_file(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": False})
        assert config_file.exists()

    def test_write_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "runtime_config.json"
        loader = FileConfigLoader(path=nested)
        loader.write({"proximal_mu": 0.01})
        assert nested.exists()

    def test_write_content_is_valid_json(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"proximal_mu": 0.005, "stop": True})
        with open(config_file) as f:
            data = json.load(f)
        assert data["proximal_mu"] == pytest.approx(0.005)
        assert data["stop"] is True

    def test_write_then_load_roundtrip(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        original = {"proximal_mu": 0.01, "pause_seconds": 5.0, "stop": False}
        loader.write(original)
        assert loader.load(round_num=1) == original

    def test_write_overwrites_previous_config(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": False, "proximal_mu": 0.01})
        loader.write({"stop": True})
        result = loader.load(round_num=1)
        assert result["stop"] is True
        assert "proximal_mu" not in result

    # ── clear ─────────────────────────────────────────────────────────────────

    def test_clear_removes_file(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        config_file.write_text(json.dumps({"stop": True}))
        loader = FileConfigLoader(path=config_file)
        loader.clear()
        assert not config_file.exists()

    def test_clear_nonexistent_file_does_not_raise(self, tmp_path):
        loader = FileConfigLoader(path=tmp_path / "nao_existe.json")
        loader.clear()

    def test_clear_then_load_returns_empty(self, tmp_path):
        config_file = tmp_path / "runtime_config.json"
        loader = FileConfigLoader(path=config_file)
        loader.write({"stop": True})
        loader.clear()
        assert loader.load(round_num=1) == {}
