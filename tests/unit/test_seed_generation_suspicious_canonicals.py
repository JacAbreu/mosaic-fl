"""
Testes para a detecção de canonicals suspeitos em generate_hsl_seed.py /
generate_bpsp_seed.py::generate_exams() — achado 2026-07-26: esses dois scripts
são o pipeline REAL usado pelos targets client-generate-seed/server-generate-seed
(o que o laptop/desktop de fato rodam), completamente separado de
integration/fapesp/loader.py (que já tinha a guarda scan_analytes/term_manager).
Sem revisão prévia como scan_analytes, o melhor que dá pra fazer aqui é avisar
alto no log — este teste confirma que o aviso dispara de ponta a ponta.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scripts.db.generate_hsl_seed as hsl_mod
import scripts.db.generate_bpsp_seed as bpsp_mod

_CSV = (
    "ID_PACIENTE|DT_COLETA|DE_ANALITO|DE_RESULTADO\n"
    "P1|01/01/2021|183|68\n"
    "P1|02/01/2021|LINFOCITOS|45\n"
)


def _fake_open(_zip_path, _entry):
    return io.StringIO(_CSV)


@pytest.mark.parametrize("mod", [hsl_mod, bpsp_mod])
class TestSuspiciousCanonicalDetection:
    def test_logs_warning_for_suspicious_canonical(self, mod, monkeypatch, caplog, tmp_path):
        monkeypatch.setattr(mod, "_open", _fake_open)
        out = io.BytesIO()

        with caplog.at_level("WARNING", logger=mod.__name__):
            mod.generate_exams(
                zip_path=tmp_path / "fake.zip", entry="fake.csv", out=out,
                patient_ids={"P1"}, att_ids=set(), alias_cache={},
            )

        assert any("canonical_suspeito_detectado" in r.message for r in caplog.records)
        assert any("'183'" in r.message for r in caplog.records)

    def test_logs_summary_at_end(self, mod, monkeypatch, caplog, tmp_path):
        monkeypatch.setattr(mod, "_open", _fake_open)
        out = io.BytesIO()

        with caplog.at_level("WARNING", logger=mod.__name__):
            mod.generate_exams(
                zip_path=tmp_path / "fake.zip", entry="fake.csv", out=out,
                patient_ids={"P1"}, att_ids=set(), alias_cache={},
            )

        assert any("RESUMO: 1 canonical" in r.message for r in caplog.records)

    def test_no_warning_when_clean(self, mod, monkeypatch, caplog, tmp_path):
        clean_csv = (
            "ID_PACIENTE|DT_COLETA|DE_ANALITO|DE_RESULTADO\n"
            "P1|01/01/2021|LINFOCITOS|45\n"
        )
        monkeypatch.setattr(mod, "_open", lambda _z, _e: io.StringIO(clean_csv))
        out = io.BytesIO()

        with caplog.at_level("WARNING", logger=mod.__name__):
            mod.generate_exams(
                zip_path=tmp_path / "fake.zip", entry="fake.csv", out=out,
                patient_ids={"P1"}, att_ids=set(), alias_cache={},
            )

        assert not any("canonical_suspeito_detectado" in r.message for r in caplog.records)
        assert not any("RESUMO:" in r.message for r in caplog.records)

    def test_does_not_duplicate_warning_across_chunks(self, mod, monkeypatch, caplog, tmp_path):
        """Mesmo canonical suspeito repetido não gera o mesmo aviso de novo —
        suspicious_seen acumula entre chunks."""
        repeated_csv = "ID_PACIENTE|DT_COLETA|DE_ANALITO|DE_RESULTADO\n"
        for i in range(5):
            repeated_csv += f"P1|01/01/2021|183|{60 + i}\n"
        monkeypatch.setattr(mod, "_open", lambda _z, _e: io.StringIO(repeated_csv))
        out = io.BytesIO()

        with caplog.at_level("WARNING", logger=mod.__name__):
            mod.generate_exams(
                zip_path=tmp_path / "fake.zip", entry="fake.csv", out=out,
                patient_ids={"P1"}, att_ids=set(), alias_cache={}, chunk_size=2,
            )

        detect_warnings = [r for r in caplog.records if "canonical_suspeito_detectado" in r.message]
        assert len(detect_warnings) == 1
