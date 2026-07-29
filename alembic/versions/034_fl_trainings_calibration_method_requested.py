"""fl_trainings — adiciona calibration_method_requested

Achado 2026-07-29: FL_CALIBRATION_METHOD estava sendo passada só no comando
`server-app` (`flwr run`), que apenas SUBMETE o treino a um SuperLink já em
execução — não é o processo que roda o ServerApp. Quem executa o ServerApp é o
próprio SuperLink (subprocess.Popen sem env=), então o valor nunca chegava lá;
o FED_CFG caía sempre no default de código ("temperature"), nunca em "auto"
como o Makefile pretendia. Corrigido movendo a variável para o alvo `superlink`
(Makefile) — mas mesmo corrigido, `calibration_per_client_json`
(fl_round_history, migration existente) grava só o MÉTODO ESCOLHIDO por cada
cliente, que é indistinguível entre "auto que escolheu temperature" e
"temperature forçado direto": as duas produzem exatamente
{"calibration_method": "temperature", "temperature": T} (ver
mosaicfl.core.client.py::_fit_local_calibrator). Sem essa coluna, a única prova
de que "auto" foi de fato usado é uma linha de log do cliente
("local_calibration_fit ... method=auto chosen=...") — fácil de perder, e o log
do ServerApp em si sai suprimido por padrão no flwr. Esta coluna grava o valor
PEDIDO (FED_CFG.calibration_method no momento do treino), auditável direto no
banco, sem depender de grep em log.

Revision ID: 034
Revises: 033
Create Date: 2026-07-29
"""
from typing import Sequence, Union

from alembic import op


revision: str = '034'
down_revision: Union[str, Sequence[str], None] = '033'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS calibration_method_requested TEXT;

        COMMENT ON COLUMN metrics.fl_trainings.calibration_method_requested IS
            '"temperature" | "isotonic" | "auto" — valor de FED_CFG.calibration_method '
            'efetivamente visto pelo ServerApp neste treino (não o método que cada '
            'cliente escolheu por baixo de "auto" — isso já está em '
            'fl_round_history.calibration_per_client_json). Existe pra distinguir '
            '"auto que escolheu temperature" de "temperature forçado direto" sem '
            'depender de log de cliente (achado 2026-07-29, ver docstring da migration).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS calibration_method_requested;
    """)
