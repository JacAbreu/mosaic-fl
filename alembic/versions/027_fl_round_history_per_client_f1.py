"""fl_round_history — adiciona per_class_f1 por cliente (antes da agregação)

Achado 2026-07-27: cada cliente já calcula seu próprio per_class_f1 local
(ver client.py::evaluate(), chave "per_class_f1_json" na resposta), mas o
servidor só persistia o valor agregado (média ponderada nativa do Flower,
metrics.fl_round_history.per_class_f1) — o valor individual de cada hospital
era descartado em aggregate_evaluate(). Sem isso, não dá pra distinguir se o
colapso de F1 em classes raras (curado_internado, melhora_pronto — ver
project_colapso_classes_raras_confirmado na memória) é um artefato só da
agregação federada (cada hospital prediz bem sua própria classe dominante,
a média é que zera) ou se já é uma falha local em cada cliente antes de
agregar. Mesmo padrão de captura de resource_per_client_json/
calibration_per_client_json (migration 025).

Revision ID: 027
Revises: 026
Create Date: 2026-07-27
"""
from typing import Sequence, Union

from alembic import op


revision: str = '027'
down_revision: Union[str, Sequence[str], None] = '026'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_round_history
            ADD COLUMN IF NOT EXISTS per_client_f1_json JSONB;

        COMMENT ON COLUMN metrics.fl_round_history.per_client_f1_json IS
            'Lista de {client_id, per_class_f1} por cliente nesta rodada, antes da '
            'agregação federada (média ponderada do Flower) — ver client.py::evaluate() '
            'e ProductionFedProxStrategy.aggregate_evaluate(). Permite distinguir colapso '
            'de F1 estrutural (já falha localmente) de colapso por diluição na agregação.';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_round_history DROP COLUMN IF EXISTS per_client_f1_json;
    """)
