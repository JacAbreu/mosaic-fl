"""simulation_node_config

Cria clinical.simulation_node_config e registra este nó como cliente HSL
na simulação de federação real (Desktop servidor + Notebook cliente).

Esta migration marca o banco como pertencente ao cliente da simulação.
Ao inspecionar o banco, o papel do nó fica explícito sem depender de
variáveis de ambiente ou documentação externa.

Contexto:
  - Papel:     client (Hospital Sírio-Libanês)
  - Servidor:  Desktop i9-13900K / 32 GB — roda BPSP
  - Cliente:   Notebook Dell Inspiron 5402 i7-1165G7 / 16 GB — roda HSL
  - Fonte:     USP-FAPESP Data Sharing COVID-19 (HSL_Janeiro2021.Zip)
  - Arquivo:   HSL_Exames_3.csv     (~1,46 M registros, 209 MB descompactado)
               HSL_Pacientes_3.csv  (~8.971 pacientes)
               HSL_Desfechos_3.csv  (~42.691 desfechos)

Revision ID: 010
Revises: 009
Create Date: 2026-06-24
"""
from typing import Sequence, Union

from alembic import op


revision: str = '010'
down_revision: Union[str, Sequence[str], None] = '009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS clinical.simulation_node_config (
            id           TEXT PRIMARY KEY DEFAULT 'current',
            node_role    TEXT NOT NULL CHECK (node_role IN ('server', 'client')),
            hospital_id  TEXT NOT NULL,
            data_source  TEXT NOT NULL DEFAULT 'FAPESP',
            description  TEXT,
            registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        COMMENT ON TABLE  clinical.simulation_node_config IS
            'Identifica o papel deste nó na simulação federada (server/client) e '
            'o hospital cujos dados estão carregados. Um único registro com id=current.';

        COMMENT ON COLUMN clinical.simulation_node_config.node_role   IS 'server = agrega pesos (Desktop BPSP) | client = treina localmente (Notebook HSL)';
        COMMENT ON COLUMN clinical.simulation_node_config.hospital_id IS 'Hospital cujos dados estão carregados neste banco (HSL | BPSP).';
        COMMENT ON COLUMN clinical.simulation_node_config.data_source IS 'Fonte dos dados clínicos: FAPESP = repositório USP-FAPESP COVID-19.';

        INSERT INTO clinical.simulation_node_config
            (id, node_role, hospital_id, data_source, description)
        VALUES (
            'current',
            'client',
            'HSL',
            'FAPESP',
            'Simulação federada TCC — Notebook Dell Inspiron 5402 (i7-1165G7 / 16 GB). '
            'Dados: Hospital Sírio-Libanês — HSL_Janeiro2021.Zip. '
            'Servidor: Desktop i9-13900K / 32 GB rodando BPSP.'
        )
        ON CONFLICT (id) DO NOTHING;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS clinical.simulation_node_config;")
