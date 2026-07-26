"""exam_records — corrige 4 canônicos curados sem nenhum registro correspondente

Investigando a lacuna de vocabulário federado (achado 2026-07-26, ver
docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md e
docs/Linha_do_Tempo_MOSAIC-FL.md), descobrimos que 4 dos 15 analitos do painel
COVID curado manualmente em knowledge.term_dictionary têm ZERO registros
correspondentes em metrics.exam_records — os dados reais usam uma grafia
diferente da registrada como canonical:

    D_DIMERO  <- DIMEROS_D_QUANTITATIVO  (13.153 registros)
    LACTATO   <- LACTATO_SANGUE          (57.835 registros)
    TGP       <- ALT_TGP                 (60.804 registros)
    TGO       <- AST_TGO                 (60.432 registros)

Esses 4 marcadores nunca contribuíram sinal nenhum pro vocabulário/treino
federado, apesar de terem sido explicitamente escolhidos pro catálogo curado.

Depois desta migration, rode scripts/compute_analyte_references.py de novo —
as referências canônicas desses 4 precisam ser recalculadas com os registros
já migrados.

Revision ID: 022
Revises: 021
Create Date: 2026-07-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = '022'
down_revision: Union[str, Sequence[str], None] = '021'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE metrics.exam_records SET analyte = 'D_DIMERO' WHERE analyte = 'DIMEROS_D_QUANTITATIVO';
        UPDATE metrics.exam_records SET analyte = 'LACTATO'  WHERE analyte = 'LACTATO_SANGUE';
        UPDATE metrics.exam_records SET analyte = 'TGP'      WHERE analyte = 'ALT_TGP';
        UPDATE metrics.exam_records SET analyte = 'TGO'      WHERE analyte = 'AST_TGO';
    """)


def downgrade() -> None:
    # Irreversível de forma limpa: depois do UPDATE, não há como distinguir quais
    # linhas de D_DIMERO/LACTATO/TGP/TGO já usavam o nome canônico originalmente
    # (nenhuma, per o achado desta migration) das que vieram da grafia antiga —
    # a migration para cá era justamente "toda linha usa a grafia antiga hoje".
    # Downgrade documentado como no-op em vez de reverter às cegas.
    pass
