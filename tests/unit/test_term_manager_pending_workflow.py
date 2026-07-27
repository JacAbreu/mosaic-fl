"""
Testes para o bloqueio de canonicals suspeitos no fluxo de revisão pré-carga
(integration/term_manager) — achado 2026-07-26: canonical="183" (Amilase do
HSL, código bruto nunca traduzido) passava batido pelo critério de "alias
verdadeiro" existente em activate_all_auto_normalized() porque
normalize("183").upper() == "183" (alias e canonical proposto idênticos,
indistinguível de uma variante de grafia legítima).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.term_manager.models import PendingTerm, ValidationResult
from integration.term_manager.pending_workflow import activate_all_auto_normalized


class TestValidationResultSuspeitos:
    def test_flags_purely_numeric_canonical(self):
        result = ValidationResult(
            total_analitos=2, resolvidos=1,
            pendentes=[
                PendingTerm(alias="183", canonical_proposto="183",
                            source="FAPESP", term_type="analyte"),
                PendingTerm(alias="Eritrocitos", canonical_proposto="ERITROCITOS",
                            source="FAPESP", term_type="analyte"),
            ],
        )
        assert [p.canonical_proposto for p in result.suspeitos] == ["183"]

    def test_no_suspicious_terms_empty_list(self):
        result = ValidationResult(
            total_analitos=1, resolvidos=0,
            pendentes=[PendingTerm(alias="Eritrocitos", canonical_proposto="ERITROCITOS",
                                    source="FAPESP", term_type="analyte")],
        )
        assert result.suspeitos == []

    def test_print_report_logs_warning_on_suspicious(self, caplog):
        result = ValidationResult(
            total_analitos=1, resolvidos=0,
            pendentes=[PendingTerm(alias="183", canonical_proposto="183",
                                    source="FAPESP", term_type="analyte")],
        )
        with caplog.at_level("WARNING", logger="integration.term_manager.models"):
            result.print_report()
        assert any("scan_analytes_suspicious_canonicals" in r.message for r in caplog.records)

    def test_print_report_no_warning_when_clean(self, caplog):
        result = ValidationResult(
            total_analitos=1, resolvidos=1, pendentes=[],
        )
        with caplog.at_level("WARNING", logger="integration.term_manager.models"):
            result.print_report()
        assert not any("scan_analytes_suspicious_canonicals" in r.message for r in caplog.records)


def _mock_conn(active_canonicals, pending_rows):
    """pending_rows: list of (alias, canonical, source) já com active=FALSE."""
    conn = MagicMock()

    def _execute(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "DISTINCT canonical" in sql:
            result.fetchall.return_value = [MagicMock(canonical=c) for c in active_canonicals]
        elif "WHERE term_type = :tt AND active = FALSE" in sql:
            result.fetchall.return_value = [
                MagicMock(alias=a, canonical=c, source=s) for a, c, s in pending_rows
            ]
        elif "SET active = TRUE" in sql and "source" in sql:
            result.rowcount = len(pending_rows)
        return result

    conn.execute.side_effect = _execute
    return conn


class TestActivateAllAutoNormalized:
    def test_blocks_on_suspicious_canonical(self, caplog):
        conn = _mock_conn(
            active_canonicals=["AMILASE"],
            pending_rows=[("183", "183", "AUTO_NORMALIZED")],
        )
        with caplog.at_level("WARNING", logger="integration.term_manager.pending_workflow"):
            n = activate_all_auto_normalized(conn)
        assert n == 0
        assert any(
            "activate_all_auto_normalized_suspicious_canonicals" in r.message
            for r in caplog.records
        )

    def test_does_not_update_db_when_blocked(self):
        conn = _mock_conn(
            active_canonicals=[],
            pending_rows=[("183", "183", "AUTO_NORMALIZED")],
        )
        activate_all_auto_normalized(conn)
        update_calls = [
            c for c in conn.execute.call_args_list
            if "UPDATE knowledge.term_dictionary" in str(c.args[0])
            and "active = TRUE" in str(c.args[0])
        ]
        assert update_calls == []

    def test_legitimate_spelling_variant_still_activates(self):
        conn = _mock_conn(
            active_canonicals=[],
            pending_rows=[("Eritrocitos", "ERITROCITOS", "AUTO_NORMALIZED")],
        )
        n = activate_all_auto_normalized(conn)
        assert n == 1  # não bloqueado — chega até o UPDATE de verdade

    def test_mixed_batch_blocks_everything_until_corrected(self):
        """Um canonical suspeito no lote bloqueia a ativação em lote inteira —
        mesmo comportamento já existente pra aliases verdadeiros/colisões."""
        conn = _mock_conn(
            active_canonicals=[],
            pending_rows=[
                ("Eritrocitos", "ERITROCITOS", "AUTO_NORMALIZED"),
                ("183", "183", "AUTO_NORMALIZED"),
            ],
        )
        n = activate_all_auto_normalized(conn)
        assert n == 0
