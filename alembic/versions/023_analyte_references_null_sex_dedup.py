"""knowledge.analyte_references — deduplica e corrige unicidade sob sex NULL

Achado 2026-07-26, durante a validação do Mecanismo B (descoberta de lacunas
de catálogo): compute_analyte_references.py::persist() e
insert_no_ref_placeholders() usam `ON CONFLICT (canonical, sex) DO UPDATE`,
mas a constraint `analyte_references_unique UNIQUE (canonical, sex)` nunca
detecta conflito quando `sex IS NULL` — semântica padrão de SQL, NULL nunca é
considerado igual a NULL para fins de unicidade. Como toda linha em uso real
tem `sex=NULL` (nenhuma referência estratificada por sexo foi implementada
ainda), TODA reexecução do script inseria uma linha nova em vez de atualizar
a existente. Achado: 2.114 linhas para 1.224 canônicos distintos — algumas
com até 5 cópias idênticas (só `computed_at` difere).

Esta migration:
  1. Deduplica: para cada `canonical` com `sex IS NULL`, mantém só a linha
     mais recente (`computed_at` mais alto), remove as demais.
  2. Cria um índice único parcial `(canonical) WHERE sex IS NULL` — a
     constraint original `(canonical, sex)` continua cobrindo os casos
     sex='M'/'F', não removida.

Depois desta migration, `ON CONFLICT (canonical) WHERE sex IS NULL` passa a
detectar conflito corretamente (ver correção em compute_analyte_references.py
no mesmo commit).

Revision ID: 023
Revises: 022
Create Date: 2026-07-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = '023'
down_revision: Union[str, Sequence[str], None] = '022'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DELETE FROM knowledge.analyte_references a
        WHERE a.sex IS NULL
          AND a.computed_at < (
              SELECT MAX(b.computed_at)
              FROM knowledge.analyte_references b
              WHERE b.canonical = a.canonical AND b.sex IS NULL
          );

        CREATE UNIQUE INDEX IF NOT EXISTS analyte_references_canonical_null_sex_uidx
            ON knowledge.analyte_references (canonical)
            WHERE sex IS NULL;
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS knowledge.analyte_references_canonical_null_sex_uidx;
    """)
    # A deduplicação de linhas não é reversível (dados removidos).
