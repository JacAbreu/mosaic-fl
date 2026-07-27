"""exam_records — corrige canonical="183" (código bruto de laboratório do HSL) para AMILASE

Investigando um caso sinalizado pela descoberta bidirecional de vocabulário
(achado 2026-07-26, ver docs/Linha_do_Tempo_MOSAIC-FL.md e página /vocab-anomalies),
confirmamos direto no CSV bruto do HSL (HSL_Exames_3.csv, dentro de
HSL_Janeiro2021.Zip) que as 2.133 linhas onde DE_EXAME="Amilase" têm
DE_ANALITO="183" — um código interno de laboratório nunca traduzido pro nome
legível. `knowledge.analyte_references` já tinha AMILASE com faixa idêntica
(28.0-100.0) à computada pra "183", com 13.189 registros legítimos do BPSP —
confirma que é o mesmo exame.

Mesmo padrão da migration 022 (D_DIMERO/LACTATO/TGP/TGO): corrige exam_records
onde ainda existir no banco (BPSP não tem nenhuma linha com analyte='183' —
UPDATE vira no-op ali, sem risco; o efeito real é no banco do HSL). Também
registra "183" como alias de AMILASE em term_dictionary — se o mesmo dado
bruto for recarregado no futuro (reload pós-ajuste), "183" resolve direto pro
canonical certo, sem precisar passar pelo fallback que criou o problema da
primeira vez.

Depois desta migration, rode scripts/compute_analyte_references.py de novo
se algum treino for gerar checkpoint novo — o vocabulário federado
(build_standard_vocab.py) já não incluirá mais "183" como canonical próprio.

Revision ID: 026
Revises: 025
Create Date: 2026-07-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = '026'
down_revision: Union[str, Sequence[str], None] = '025'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE metrics.exam_records SET analyte = 'AMILASE' WHERE analyte = '183';

        DELETE FROM knowledge.term_dictionary
         WHERE term_type = 'analyte' AND canonical = '183';

        INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source, active)
        VALUES ('analyte', 'AMILASE', '183', 'MANUAL_CORRECTION', TRUE)
        ON CONFLICT (term_type, canonical, alias) DO NOTHING;

        DELETE FROM knowledge.analyte_references WHERE canonical = '183';
    """)


def downgrade() -> None:
    # Irreversível de forma limpa, mesmo motivo da migration 022: depois do UPDATE
    # não há como distinguir linhas que já eram AMILASE de linhas que vieram de
    # "183" (todo o "183" existente virou AMILASE, sem exceção). Downgrade
    # documentado como no-op em vez de reverter às cegas.
    pass
