"""
Testes para scripts/compute_analyte_references.py.

Sem fixture Postgres real (nenhum dos scripts de manutenção do projeto tem —
ver reset_data.py, auditar_classes_atuais.py) — mocka na fronteira
conn.execute()/conn.commit(), testando a lógica de decisão em Python.
Funções puras (classify()) ganham testes tabulares diretos, sem mock.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.compute_analyte_references import (
    backfill_classifications,
    classify,
    compute,
    insert_no_ref_placeholders,
    persist,
)


class TestClassify:
    def test_ref_low_none_is_no_ref(self):
        assert classify(10.0, None, 20.0) == "NO_REF"

    def test_ref_high_none_is_no_ref(self):
        assert classify(10.0, 5.0, None) == "NO_REF"

    def test_both_zero_is_no_ref(self):
        assert classify(10.0, 0.0, 0.0) == "NO_REF"

    def test_value_below_ref_low_is_low(self):
        assert classify(3.0, 5.0, 20.0) == "LOW"

    def test_value_above_ref_high_is_high(self):
        assert classify(25.0, 5.0, 20.0) == "HIGH"

    def test_value_inside_range_is_normal(self):
        assert classify(10.0, 5.0, 20.0) == "NORMAL"

    def test_value_exactly_at_ref_low_is_normal(self):
        assert classify(5.0, 5.0, 20.0) == "NORMAL"

    def test_value_exactly_at_ref_high_is_normal(self):
        assert classify(20.0, 5.0, 20.0) == "NORMAL"


class TestCompute:
    def test_rounds_to_four_decimals_and_sets_source(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(canonical="PCR", n_hospitals=2, ref_low=1.23456, ref_high=9.87654),
        ]
        entries = compute(conn, min_hospitals=2)
        assert entries == [{
            "canonical": "PCR", "sex": None,
            "ref_low": 1.2346, "ref_high": 9.8765,
            "n_hospitals": 2, "source": "MEDIA_HOSPITAIS_PARTICIPANTES",
        }]

    def test_passes_min_hospitals_as_bound_param(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        compute(conn, min_hospitals=3)
        _, kwargs_or_params = conn.execute.call_args[0]
        assert kwargs_or_params == {"min_h": 3}

    def test_empty_result(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        assert compute(conn) == []


class TestPersist:
    def test_calls_execute_once_per_entry_and_commits(self):
        conn = MagicMock()
        entries = [
            {"canonical": "PCR", "sex": None, "ref_low": 1.0, "ref_high": 2.0,
             "n_hospitals": 2, "source": "MEDIA_HOSPITAIS_PARTICIPANTES"},
            {"canonical": "TGP", "sex": None, "ref_low": 3.0, "ref_high": 4.0,
             "n_hospitals": 1, "source": "MEDIA_HOSPITAIS_PARTICIPANTES"},
        ]
        n = persist(conn, entries)
        assert n == 2
        assert conn.execute.call_count == 2
        conn.commit.assert_called_once()

    def test_empty_entries_still_commits(self):
        conn = MagicMock()
        assert persist(conn, []) == 0
        conn.commit.assert_called_once()


class TestInsertNoRefPlaceholders:
    def test_dry_run_does_not_commit(self):
        conn = MagicMock()
        conn.execute.return_value.scalar_one.return_value = 526
        result = insert_no_ref_placeholders(conn, dry_run=True)
        assert result == []
        conn.commit.assert_not_called()

    def test_returns_inserted_canonicals_and_commits(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(canonical="PCR"),
            SimpleNamespace(canonical="TTPA_NORMAL_DO_DIA"),
        ]
        result = insert_no_ref_placeholders(conn)
        assert result == ["PCR", "TTPA_NORMAL_DO_DIA"]
        conn.commit.assert_called_once()

    def test_no_candidates_returns_empty_list(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        assert insert_no_ref_placeholders(conn) == []
        conn.commit.assert_called_once()


class TestBackfillClassifications:
    def test_returns_rowcount_and_commits(self):
        conn = MagicMock()
        conn.execute.return_value.rowcount = 865552
        updated = backfill_classifications(conn)
        assert updated == 865552
        conn.commit.assert_called_once()

    def test_zero_rowcount(self):
        conn = MagicMock()
        conn.execute.return_value.rowcount = 0
        assert backfill_classifications(conn) == 0
