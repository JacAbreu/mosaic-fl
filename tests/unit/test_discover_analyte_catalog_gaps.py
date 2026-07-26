"""
Testes para scripts/discover_analyte_catalog_gaps.py — descoberta de analitos
fora do catálogo de 15 curados manualmente (achado 2026-07-26: 819 analitos
distintos nos dados reais, ~82% dos registros do BPSP fora do catálogo ativo,
virando <UNK> na tokenização em vez de ganhar um token próprio).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.discover_analyte_catalog_gaps import (
    find_candidates,
    get_active_canonicals,
    insert_candidates,
    select_insertable,
    warn_possible_synonyms,
)


def _candidate(analyte, ref_source, ref_low=None, ref_high=None, n_hospitals=1, n_records=1000):
    return {
        "analyte": analyte, "n_hospitals": n_hospitals, "n_records": n_records,
        "has_real_ref": (
            ref_source == "MEDIA_HOSPITAIS_PARTICIPANTES"
            and ref_low is not None and not (ref_low == 0.0 and ref_high == 0.0)
        ),
        "ref_source": ref_source,
    }


class TestFindCandidates:
    def test_computes_has_real_ref_for_valid_reference(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(analyte="LINFOCITOS", n_hospitals=1, n_records=254614,
                             ref_low=15.0, ref_high=45.0, ref_source="MEDIA_HOSPITAIS_PARTICIPANTES"),
        ]
        result = find_candidates(conn)
        assert result[0]["has_real_ref"] is True

    def test_no_ref_placeholder_source_is_not_real_ref(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(analyte="PCR", n_hospitals=1, n_records=97918,
                             ref_low=0.0, ref_high=0.0, ref_source="SEM_REFERENCIA_INSTITUCIONAL"),
        ]
        result = find_candidates(conn)
        assert result[0]["has_real_ref"] is False

    def test_no_reference_row_at_all_is_not_real_ref(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(analyte="ALGUM_EXAME_RARO", n_hospitals=1, n_records=150,
                             ref_low=None, ref_high=None, ref_source=None),
        ]
        result = find_candidates(conn)
        assert result[0]["has_real_ref"] is False

    def test_passes_thresholds_as_bound_params(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        find_candidates(conn, min_records=50, min_hospitals=2)
        _, params = conn.execute.call_args[0]
        assert params == {"min_records": 50, "min_hospitals": 2}


class TestSelectInsertable:
    def test_default_tier_excludes_no_ref(self):
        candidates = [
            _candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0),
            _candidate("PCR_SEM_REF", "SEM_REFERENCIA_INSTITUCIONAL", 0.0, 0.0),
        ]
        selected = select_insertable(candidates)
        assert [c["analyte"] for c in selected] == ["LINFOCITOS"]

    def test_include_no_ref_returns_everything(self):
        candidates = [
            _candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0),
            _candidate("PCR_SEM_REF", "SEM_REFERENCIA_INSTITUCIONAL", 0.0, 0.0),
        ]
        selected = select_insertable(candidates, include_no_ref=True)
        assert len(selected) == 2

    def test_empty_candidates(self):
        assert select_insertable([]) == []


class TestWarnPossibleSynonyms:
    def test_detects_substring_match(self):
        candidates = [_candidate("ALT_TGP", "MEDIA_HOSPITAIS_PARTICIPANTES", 5.0, 40.0)]
        warnings = warn_possible_synonyms(candidates, {"TGP"})
        assert len(warnings) == 1
        assert "ALT_TGP" in warnings[0]

    def test_known_limitation_misses_non_substring_synonym(self):
        """Documenta a limitação, não um comportamento desejado — DIMEROS_D_QUANTITATIVO
        vs D_DIMERO não compartilham substring, a heurística não pega esse caso
        (por isso a migration 022_fix_dead_curated_canonicals corrigiu manualmente)."""
        candidates = [_candidate("DIMEROS_D_QUANTITATIVO", "MEDIA_HOSPITAIS_PARTICIPANTES", 0.1, 0.5)]
        warnings = warn_possible_synonyms(candidates, {"D_DIMERO"})
        assert warnings == []

    def test_no_active_canonicals_no_warnings(self):
        candidates = [_candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0)]
        assert warn_possible_synonyms(candidates, set()) == []


class TestGetActiveCanonicals:
    def test_returns_set_of_canonicals(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            SimpleNamespace(canonical="PCR"), SimpleNamespace(canonical="TGP"),
        ]
        assert get_active_canonicals(conn) == {"PCR", "TGP"}


class TestInsertCandidates:
    def test_canonical_equals_alias_and_source_auto_discovered(self):
        conn = MagicMock()
        candidates = [_candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0)]
        n = insert_candidates(conn, candidates)
        assert n == 1
        _, params = conn.execute.call_args[0]
        assert params == {"canonical": "LINFOCITOS"}
        conn.commit.assert_called_once()

    def test_on_conflict_reactivates_instead_of_do_nothing(self):
        """Sem isso, reinserir um canonical já existente mas desativado
        (active=FALSE) não teria efeito nenhum — build_standard_vocab() nunca
        voltaria a incluí-lo, mesmo com insert_candidates() "funcionando"."""
        conn = MagicMock()
        insert_candidates(conn, [_candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0)])
        sql_text = str(conn.execute.call_args[0][0])
        assert "DO UPDATE SET active = TRUE" in sql_text
        assert "DO NOTHING" not in sql_text

    def test_one_insert_per_candidate(self):
        conn = MagicMock()
        candidates = [
            _candidate("LINFOCITOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 15.0, 45.0),
            _candidate("NEUTROFILOS", "MEDIA_HOSPITAIS_PARTICIPANTES", 40.0, 70.0),
        ]
        n = insert_candidates(conn, candidates)
        assert n == 2
        assert conn.execute.call_count == 2

    def test_empty_candidates_still_commits(self):
        conn = MagicMock()
        assert insert_candidates(conn, []) == 0
        conn.commit.assert_called_once()
