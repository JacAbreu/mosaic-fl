"""fl_round_history — adiciona detalhamento por cliente (recurso e calibração)

Achado 2026-07-26: o custo computacional por cliente (energia GPU/CPU/RAM antes
da agregação — resource_per_client_json, ver aggregate_resource_metrics em
federated.py) e a calibração individual de cada hospital antes da agregação
federada (temperature/isotonic por cliente) só existiam em log
(logger.info("round_resources"...), clientapp_<id>_<timestamp>.log) — nunca
persistidos em nenhuma tabela. Só o agregado (pico/média entre clientes, T
federado final) ia pro banco.

Junto com esta migration, ProductionFedProxStrategy passa a chamar
save_round_history() a CADA rodada (não só uma vez no fim do treino) — fecha
também o gap de que, antes desta correção, o histórico rodada-a-rodada só
ficava garantido em banco se o treino chegasse até completed_training()
(um crash no meio perdia tudo exceto os arquivos logs/round_N_metrics.json).

Revision ID: 025
Revises: 024
Create Date: 2026-07-26
"""
from typing import Sequence, Union

from alembic import op


revision: str = '025'
down_revision: Union[str, Sequence[str], None] = '024'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_round_history
            ADD COLUMN IF NOT EXISTS resource_per_client_json JSONB;
        ALTER TABLE metrics.fl_round_history
            ADD COLUMN IF NOT EXISTS calibration_per_client_json JSONB;

        COMMENT ON COLUMN metrics.fl_round_history.resource_per_client_json IS
            'Lista de {duration_s, cpu_pct, ram_mb, gpu_power_w?, gpu_energy_wh?} por '
            'cliente nesta rodada, antes da agregação — ver aggregate_resource_metrics '
            'em mosaicfl.core.federated. NULL quando FL_COLLECT_RESOURCE_METRICS=0.';
        COMMENT ON COLUMN metrics.fl_round_history.calibration_per_client_json IS
            'Lista de {client_id, method, temperature?} por cliente nesta rodada, antes '
            'da agregação federada — só presente na rodada em que o servidor pediu '
            'calibrate=True (ver FedProxClient._fit_local_calibrator).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_round_history DROP COLUMN IF EXISTS calibration_per_client_json;
        ALTER TABLE metrics.fl_round_history DROP COLUMN IF EXISTS resource_per_client_json;
    """)
