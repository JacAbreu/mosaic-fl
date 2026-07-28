"""clinical.fl_orchestration_config — adiciona class_weight_overrides_json

Achado 2026-07-27 (mesma discussão da seção 14 de docs/pesquisa_baseline_implementacao_
fontes_bibliograficas.md): peso de classe explícito (cost-sensitive learning, ver
mosaicfl.core.class_weighting) precisa ser IDÊNTICO nos dois hospitais — BPSP e HSL têm
bancos locais separados (dado clínico nunca sai do hospital, por desenho), então um
".env" por máquina exigiria sincronizar manualmente o mesmo valor nos dois lados, sem
nenhuma garantia. O canal certo é o mesmo já usado por proximal_mu/pause_seconds/stop:
o servidor (SuperLink, roda no desktop) lê clinical.fl_orchestration_config a cada
rodada e empurra pro cliente via config da rodada (mesmo padrão de vocab_json, ver
fit_config_mixin.py) — um valor só, editável ao vivo, sem tocar .env de nenhuma máquina.

Revision ID: 028
Revises: 027
Create Date: 2026-07-27
"""
from typing import Sequence, Union

from alembic import op


revision: str = '028'
down_revision: Union[str, Sequence[str], None] = '027'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.fl_orchestration_config
            ADD COLUMN IF NOT EXISTS class_weight_overrides_json JSONB;

        COMMENT ON COLUMN clinical.fl_orchestration_config.class_weight_overrides_json IS
            'Peso explícito por classe (cost-sensitive learning, Strategy pattern em '
            'mosaicfl.core.class_weighting) — objeto {nome_da_classe: peso}. NULL = '
            'nenhuma classe usa peso explícito, todas caem em class_balanced (frequência '
            'local, comportamento padrão do projeto). Lido a cada rodada por '
            'configure_fit()/configure_evaluate() e empurrado pro cliente via config, '
            'idêntico nos dois hospitais — ver docs/pesquisa_baseline_implementacao_'
            'fontes_bibliograficas.md, seção 14.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE clinical.fl_orchestration_config DROP COLUMN IF EXISTS class_weight_overrides_json;
    """)
