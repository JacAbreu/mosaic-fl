"""
Teste de integração contra Postgres REAL — confirma que insert_candidates()
(scripts/discover_analyte_catalog_gaps.py) reativa um canonical desativado em
vez de silenciosamente não fazer nada.

Achado 2026-07-26, discutindo a validação do vocabulário federado bidirecional:
o INSERT original usava `ON CONFLICT (term_type, canonical, alias) DO NOTHING`
— se um canonical já existia na tabela com active=FALSE, reinseri-lo batia no
conflito e não fazia nada, deixando o registro desativado pra sempre (mesmo com
insert_candidates() reportando sucesso). build_standard_vocab() filtra
`WHERE active = TRUE`, então o analito nunca voltaria a aparecer no vocab
federado. Corrigido para `DO UPDATE SET active = TRUE`.

IMPORTANTE: diferente de test_analyte_references_unique_index_live.py, este
teste NÃO usa o padrão trans.rollback() — insert_candidates() chama
conn.commit() internamente, o que desassocia a transação externa e faz o
rollback() do fixture virar no-op (achado ao vivo: um rollback "bem-sucedido"
deixou uma linha de teste vazando na tabela real por alguns segundos, limpa
manualmente depois). Em vez disso, cada teste limpa explicitamente sua própria
linha em um finally — determinístico, não depende de semântica de transação
aninhada do SQLAlchemy.

Requer FL_TEST_DB_URL (não FL_DB_URL) definido explicitamente — não roda por
padrão.

Uso:
    FL_TEST_DB_URL=postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl \\
        pytest tests/integration/test_insert_candidates_reactivation_live.py -v
"""
import os

import pytest

_TEST_DB_URL = os.getenv("FL_TEST_DB_URL")

pytestmark = pytest.mark.skipif(
    not _TEST_DB_URL,
    reason="requer FL_TEST_DB_URL apontando para um Postgres real e descartável — "
           "não roda por padrão (evita exigir banco disponível na suíte padrão)",
)

_TEST_CANONICAL = "__TEST_INSERT_CANDIDATES_REACTIVATION__"


@pytest.fixture
def live_conn():
    import sqlalchemy as sa
    engine = sa.create_engine(_TEST_DB_URL)
    with engine.connect() as conn:
        try:
            yield conn
        finally:
            # Limpeza explícita, não rollback — insert_candidates() já commitou
            # internamente, não há transação pendente pra reverter.
            conn.execute(sa.text("""
                DELETE FROM knowledge.term_dictionary
                WHERE term_type = 'analyte' AND canonical = :c AND alias = :c
            """), {"c": _TEST_CANONICAL})
            conn.commit()


class TestInsertCandidatesReactivationLive:
    def test_reactivates_existing_deactivated_canonical(self, live_conn):
        import sqlalchemy as sa
        from scripts.discover_analyte_catalog_gaps import insert_candidates

        live_conn.execute(sa.text("""
            INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source, active)
            VALUES ('analyte', :c, :c, 'AUTO_DISCOVERED', FALSE)
        """), {"c": _TEST_CANONICAL})
        live_conn.commit()

        n = insert_candidates(live_conn, [{"analyte": _TEST_CANONICAL}])
        assert n == 1

        row = live_conn.execute(sa.text("""
            SELECT active FROM knowledge.term_dictionary
            WHERE term_type = 'analyte' AND canonical = :c AND alias = :c
        """), {"c": _TEST_CANONICAL}).fetchone()
        assert row.active is True

    def test_does_not_duplicate_row_on_reactivation(self, live_conn):
        import sqlalchemy as sa
        from scripts.discover_analyte_catalog_gaps import insert_candidates

        live_conn.execute(sa.text("""
            INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source, active)
            VALUES ('analyte', :c, :c, 'AUTO_DISCOVERED', FALSE)
        """), {"c": _TEST_CANONICAL})
        live_conn.commit()

        insert_candidates(live_conn, [{"analyte": _TEST_CANONICAL}])

        rows = live_conn.execute(sa.text("""
            SELECT id FROM knowledge.term_dictionary
            WHERE term_type = 'analyte' AND canonical = :c AND alias = :c
        """), {"c": _TEST_CANONICAL}).fetchall()
        assert len(rows) == 1

    def test_fresh_insert_when_no_prior_row(self, live_conn):
        import sqlalchemy as sa
        from scripts.discover_analyte_catalog_gaps import insert_candidates

        n = insert_candidates(live_conn, [{"analyte": _TEST_CANONICAL}])
        assert n == 1

        row = live_conn.execute(sa.text("""
            SELECT active, source FROM knowledge.term_dictionary
            WHERE term_type = 'analyte' AND canonical = :c AND alias = :c
        """), {"c": _TEST_CANONICAL}).fetchone()
        assert row.active is True
        assert row.source == "AUTO_DISCOVERED"
