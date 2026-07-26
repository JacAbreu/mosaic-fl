"""
Teste de integração contra Postgres REAL (não mockado) — confirma que o índice
único parcial `analyte_references_canonical_null_sex_uidx` (migration 023) bloqueia
duplicata mesmo bypassando completamente o código Python da aplicação (INSERT cru).

Achado 2026-07-26: knowledge.analyte_references acumulava linhas duplicadas a cada
reexecução de compute_analyte_references.py porque `ON CONFLICT (canonical, sex)`
nunca detecta conflito quando sex IS NULL (semântica padrão de SQL). Os testes
unitários de compute_analyte_references.py mockam conn.execute()/conn.commit() — não
têm como verificar o comportamento real do índice, só que o Python monta a chamada
certa. Este teste fecha essa lacuna, mas só roda contra um banco descartável — nunca
o de produção (ver FL_TEST_DB_URL abaixo).

NUNCA COMMITA — todo o teste roda dentro de uma transação com rollback explícito no
finally, mesmo em caso de sucesso. Seguro rodar contra qualquer banco, mas ainda
assim requer FL_TEST_DB_URL (não FL_DB_URL) definido explicitamente — não roda por
padrão em `make test`/CI, evita exigir Postgres disponível pra suíte padrão. Primeiro
teste de integração contra banco real deste projeto (sem precedente anterior) —
ver docs/Linha_do_Tempo_MOSAIC-FL.md, 2026-07-26.

Uso:
    FL_TEST_DB_URL=postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl \\
        pytest tests/integration/test_analyte_references_unique_index_live.py -v
"""
import os

import pytest

_TEST_DB_URL = os.getenv("FL_TEST_DB_URL")

pytestmark = pytest.mark.skipif(
    not _TEST_DB_URL,
    reason="requer FL_TEST_DB_URL apontando para um Postgres real e descartável — "
           "não roda por padrão (evita exigir banco disponível na suíte padrão)",
)

_TEST_CANONICAL = "__TEST_UNIQUE_INDEX_LIVE__"


@pytest.fixture
def live_conn():
    import sqlalchemy as sa
    engine = sa.create_engine(_TEST_DB_URL)
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            yield conn
        finally:
            trans.rollback()  # nunca persiste, mesmo em sucesso — seguro contra qualquer banco


class TestAnalyteReferencesUniqueIndexLive:
    def test_duplicate_null_sex_insert_is_rejected(self, live_conn):
        import sqlalchemy as sa
        from sqlalchemy.exc import IntegrityError

        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, NULL, 1.0, 2.0, 1, 'TESTE_INTEGRACAO', NOW())
        """), {"c": _TEST_CANONICAL})

        with pytest.raises(IntegrityError, match="analyte_references_canonical_null_sex_uidx"):
            live_conn.execute(sa.text("""
                INSERT INTO knowledge.analyte_references
                    (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
                VALUES (:c, NULL, 3.0, 4.0, 1, 'TESTE_INTEGRACAO', NOW())
            """), {"c": _TEST_CANONICAL})

    def test_different_sex_values_do_not_conflict(self, live_conn):
        """sex='M' e sex='F' continuam livres pra coexistir com sex=NULL — só o
        NULL precisava do índice parcial extra (a constraint original (canonical,
        sex) já cobria os casos com sex não-nulo corretamente)."""
        import sqlalchemy as sa

        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, NULL, 1.0, 2.0, 1, 'TESTE_INTEGRACAO', NOW())
        """), {"c": _TEST_CANONICAL})
        # não deve levantar exceção — sex diferente, mesmo canonical
        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, 'M', 1.0, 2.0, 1, 'TESTE_INTEGRACAO', NOW())
        """), {"c": _TEST_CANONICAL})
        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, 'F', 1.0, 2.0, 1, 'TESTE_INTEGRACAO', NOW())
        """), {"c": _TEST_CANONICAL})

    def test_on_conflict_upsert_works_correctly(self, live_conn):
        """Confirma que ON CONFLICT (canonical) WHERE sex IS NULL (a correção
        aplicada em compute_analyte_references.py) faz UPDATE em vez de duplicar."""
        import sqlalchemy as sa

        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, NULL, 1.0, 2.0, 1, 'TESTE_INTEGRACAO', NOW())
            ON CONFLICT (canonical) WHERE sex IS NULL DO UPDATE SET ref_low = EXCLUDED.ref_low
        """), {"c": _TEST_CANONICAL})
        live_conn.execute(sa.text("""
            INSERT INTO knowledge.analyte_references
                (canonical, sex, ref_low, ref_high, n_hospitals, source, computed_at)
            VALUES (:c, NULL, 9.0, 9.0, 1, 'TESTE_INTEGRACAO', NOW())
            ON CONFLICT (canonical) WHERE sex IS NULL DO UPDATE SET ref_low = EXCLUDED.ref_low
        """), {"c": _TEST_CANONICAL})

        rows = live_conn.execute(sa.text(
            "SELECT ref_low FROM knowledge.analyte_references WHERE canonical=:c AND sex IS NULL"
        ), {"c": _TEST_CANONICAL}).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 9.0
