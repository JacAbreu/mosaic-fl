import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.preprocessor import split_by_institution


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


class TestSplitByInstitution:

    def test_creates_correct_number_of_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        assert len(clients) == 3

    def test_all_rows_preserved(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        total = sum(len(df) for df in clients.values())
        assert total == len(sample_df)

    def test_no_overlap_between_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        indices = [set(df.index) for df in clients.values()]
        for i, s1 in enumerate(indices):
            for j, s2 in enumerate(indices):
                if i != j:
                    assert len(s1 & s2) == 0

    def test_stratify_col_runs_without_error(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5, stratify_col="desfecho")
        assert len(clients) >= 1

    def test_random_state_reproducible(self, sample_df):
        c1 = split_by_institution(sample_df, num_clients=5, random_state=42)
        c2 = split_by_institution(sample_df, num_clients=5, random_state=42)
        for cid in c1:
            pd.testing.assert_frame_equal(c1[cid].reset_index(drop=True),
                                           c2[cid].reset_index(drop=True))

    def test_num_clients_capped_by_institutions(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=100)
        assert len(clients) == 3
