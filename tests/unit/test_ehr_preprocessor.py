import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.preprocessor import EHRPreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instituicao":   ["HospA", "HospA", "HospB", "HospB", "HospC"],
        "idade":         [25.0, 6.0, 45.0, 365.0, 70.0],
        "idade_unidade": ["anos", "meses", "anos", "dias", "anos"],
        "peso":          [150.0, 70.0, 180.0, 50.0, 80.0],
        "peso_unidade":  ["lb", "kg", "lb", "kg", "kg"],
        "sintoma":       ["febre", "tosse", "dispneia", "fadiga", "mialgia"],
        "exame":         ["rt_pcr_positivo", "tomografia_normal", "rx_consolidacao",
                          "pcr_negativo", "tomografia_vidro_fosco"],
        "diagnostico":   ["covid19_leve", "covid19_moderado", "pneumonia_bacteriana",
                          "covid19_grave", "alta"],
        "desfecho":      [0, 0, 1, 1, 0],
    })


class TestEHRPreprocessor:

    def test_normalize_units_converts_months(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[1, "idade"] - 0.5) < 0.01

    def test_normalize_units_converts_days(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[3, "idade"] - 1.0) < 0.05

    def test_normalize_units_sets_unidade_anos(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert (df["idade_unidade"] == "anos").all()

    def test_normalize_units_converts_lb_to_kg(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[0, "peso"] - 150 * 0.453592) < 0.01
        assert abs(df.loc[2, "peso"] - 180 * 0.453592) < 0.01

    def test_normalize_units_preserves_kg(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert df.loc[1, "peso"] == 70.0

    def test_clean_text_lowercase(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"sintoma": ["FEBRE", "Tosse"]})
        result = pre.clean_text(df, ["sintoma"])
        assert all(v == v.lower() for v in result["sintoma"])

    def test_clean_text_removes_special_chars(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"sintoma": ["febre!@#$%", "tosse&*("]})
        result = pre.clean_text(df, ["sintoma"])
        assert "!" not in result["sintoma"].iloc[0]
        assert "&" not in result["sintoma"].iloc[1]

    def test_build_vocabulary_special_tokens(self, sample_df):
        pre = EHRPreprocessor()
        pre.build_vocabulary(sample_df, ["sintoma", "exame", "diagnostico"])
        for token in ["<PAD>", "<UNK>", "<MASK>", "<CLS>"]:
            assert token in pre.vocab_map
        assert pre.vocab_map["<PAD>"] == 0

    def test_build_vocabulary_no_duplicates(self, sample_df):
        pre = EHRPreprocessor()
        vocab = pre.build_vocabulary(sample_df, ["sintoma"])
        values = list(vocab.values())
        assert len(values) == len(set(values))

    def test_handle_missing_imputes_median(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0]})
        result = pre.handle_missing(df.copy())
        assert result.loc[2, "num"] == 2.0

    def test_handle_missing_imputes_unk(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"cat": ["a", np.nan, "c"]})
        result = pre.handle_missing(df.copy())
        assert result.loc[1, "cat"] == "<UNK>"

    def test_process_returns_df_and_summary(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, summary = pre.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        assert isinstance(df_proc, pd.DataFrame)
        assert isinstance(summary, dict)
        assert summary["total_amostras"] == len(sample_df)
        assert summary["tamanho_vocabulario"] > 4

    def test_process_creates_encoded_columns(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma"])
        assert "sintoma_encoded" in df_proc.columns
        assert df_proc["sintoma_encoded"].dtype in [int, np.int64, np.int32]

    def test_process_reject_count_zero_on_clean_data(self, sample_df):
        pre = EHRPreprocessor()
        _, summary = pre.process(sample_df, text_cols=["sintoma"])
        assert summary["amostras_rejeitadas"] == 0

    def test_transform_log_populated(self, sample_df):
        pre = EHRPreprocessor()
        _, summary = pre.process(sample_df, text_cols=["sintoma"])
        assert len(summary["transformacoes"]) > 0
        steps = [t["step"] for t in summary["transformacoes"]]
        assert "clean" in steps
        assert "vocab" in steps
