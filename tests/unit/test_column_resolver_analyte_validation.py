"""
Testes para looks_like_valid_analyte_name() (integration/column_resolver.py) —
heurística compartilhada entre a revisão pré-carga (term_manager) e a página
pós-hoc /vocab-anomalies (admin.py). Achado 2026-07-26: canonical="183" (código
interno de laboratório do HSL pra Amilase, nunca traduzido) passava batido pelo
critério de "alias verdadeiro" existente (normalize("183").upper() == "183").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.column_resolver import looks_like_valid_analyte_name


class TestLooksLikeValidAnalyteName:
    def test_rejects_purely_numeric(self):
        assert looks_like_valid_analyte_name("183") is False

    def test_rejects_purely_numeric_long(self):
        assert looks_like_valid_analyte_name("1234567890") is False

    def test_accepts_real_analyte_name(self):
        assert looks_like_valid_analyte_name("AMILASE") is True

    def test_accepts_name_with_underscores(self):
        assert looks_like_valid_analyte_name("PROTEINA_C_REATIVA") is True

    def test_accepts_alphanumeric_mix(self):
        """Códigos alfanuméricos (não puramente numéricos) não são bloqueados —
        heurística conservadora, só pega o caso confirmado (puramente numérico)."""
        assert looks_like_valid_analyte_name("COVID19_ANTICORPOS_IGG") is True

    def test_empty_string_rejected_as_invalid(self):
        assert looks_like_valid_analyte_name("") is False
