"""
Testes para scripts/build_standard_vocab.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.build_standard_vocab import _make_token, _SPECIAL, build_standard_vocab


class TestMakeToken:
    def test_analyte_only_mode_ignores_classification(self):
        assert _make_token("PCR", "HIGH", mode="ANALYTE_ONLY") == "PCR"

    def test_class_only_mode_ignores_analyte(self):
        assert _make_token("PCR", "HIGH", mode="CLASS_ONLY") == "HIGH"

    def test_full_mode_no_ref_uses_analyte_alone(self):
        assert _make_token("PCR", "NO_REF", mode="FULL") == "PCR"

    def test_full_mode_with_classification_combines(self):
        assert _make_token("PCR", "HIGH", mode="FULL") == "PCR_HIGH"

    def test_default_mode_is_full(self):
        assert _make_token("LEUCOCITOS", "NORMAL") == "LEUCOCITOS_NORMAL"


def _mock_engine_with_rows(rows):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine


class TestBuildStandardVocab:
    def test_analyte_with_valid_refs_gets_three_tokens(self):
        rows = [SimpleNamespace(canonical="PCR", ref_low=5.0, ref_high=10.0)]
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab = build_standard_vocab("postgresql://fake")
        assert "PCR_HIGH" in vocab
        assert "PCR_NORMAL" in vocab
        assert "PCR_LOW" in vocab
        assert len(vocab) == len(_SPECIAL) + 3

    def test_analyte_without_refs_gets_one_no_ref_token(self):
        rows = [SimpleNamespace(canonical="RARO", ref_low=None, ref_high=None)]
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab = build_standard_vocab("postgresql://fake")
        assert vocab["RARO"] == len(_SPECIAL)
        assert len(vocab) == len(_SPECIAL) + 1

    def test_zero_refs_treated_as_no_ref(self):
        rows = [SimpleNamespace(canonical="SEM_REF_INSTITUCIONAL", ref_low=0.0, ref_high=0.0)]
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab = build_standard_vocab("postgresql://fake")
        assert len(vocab) == len(_SPECIAL) + 1

    def test_special_tokens_present_at_fixed_ids(self):
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows([])):
            vocab = build_standard_vocab("postgresql://fake")
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1
        assert vocab["<CLS>"] == 2

    def test_ids_are_sequential_with_no_gaps(self):
        rows = [
            SimpleNamespace(canonical="A", ref_low=1.0, ref_high=2.0),
            SimpleNamespace(canonical="B", ref_low=None, ref_high=None),
        ]
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab = build_standard_vocab("postgresql://fake")
        ids = sorted(vocab.values())
        assert ids == list(range(len(vocab)))

    def test_deterministic_across_calls(self):
        rows = [
            SimpleNamespace(canonical="A", ref_low=1.0, ref_high=2.0),
            SimpleNamespace(canonical="B", ref_low=None, ref_high=None),
        ]
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab1 = build_standard_vocab("postgresql://fake")
        with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
            vocab2 = build_standard_vocab("postgresql://fake")
        assert vocab1 == vocab2


class TestVocabOverflowGuard:
    def test_exits_when_vocab_exceeds_configured_size(self, monkeypatch, capsys):
        import scripts.build_standard_vocab as mod
        from mosaicfl.core.config import MODEL_CFG

        # MODEL_CFG é um dataclass frozen — object.__setattr__ contorna isso só
        # para o teste, sempre restaurado no finally (singleton de módulo).
        object.__setattr__(MODEL_CFG, "vocab_size", 3)
        try:
            monkeypatch.setattr(sys, "argv", ["build_standard_vocab.py", "--db-url", "postgresql://fake", "--dry-run"])
            rows = [SimpleNamespace(canonical="A", ref_low=1.0, ref_high=2.0)]
            with patch("scripts.build_standard_vocab.create_engine", return_value=_mock_engine_with_rows(rows)):
                with pytest.raises(SystemExit) as exc_info:
                    mod.main()
            assert exc_info.value.code == 1
        finally:
            object.__setattr__(MODEL_CFG, "vocab_size", 10_000)
