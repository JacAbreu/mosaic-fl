"""fl_trainings — adiciona convergence_round

Achado 2026-08-01, avaliando o treino 85 (primeiro treino real com
FL_EARLY_STOP=true funcionando de ponta a ponta): não existia nenhuma coluna
persistindo em QUAL rodada a convergência foi detectada — só o booleano
`converged`. A única fonte era o log do ServerApp (`convergence_detected
round=N`), que já sabemos (achado 2026-07-29) sair suprimido do terminal por
padrão e viver só em `experiments/logs/serverapp_*.log`, fácil de perder.

Importante: NÃO é o mesmo valor que `ProductionFedProxStrategy.tracker.
converged_round` (`mosaicfl.core.convergence.ConvergenceTracker`) — esse é
`len(self.history)`, um índice interno da janela deslizante que só cresce a
partir de `min_rounds+1` (bug real que quebrou a primeira versão do early
stop real, ver `infrastructure/mosaicfl_server/strategy/core.py`, comentário
em `aggregate_evaluate`). Esta coluna grava o `server_round` de verdade — o
número real da rodada do Flower em que `converged` virou `True` pela primeira
vez — para exibição confiável em `/fl-training-results` (best_round ×
convergence_round) sem essa ambiguidade.

Revision ID: 035
Revises: 034
Create Date: 2026-08-02
"""
from typing import Sequence, Union

from alembic import op


revision: str = '035'
down_revision: Union[str, Sequence[str], None] = '034'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings
            ADD COLUMN IF NOT EXISTS convergence_round INTEGER;

        COMMENT ON COLUMN metrics.fl_trainings.convergence_round IS
            'Rodada real do Flower (server_round) em que a convergência foi '
            'detectada pela primeira vez — NULL se o treino nunca convergiu. '
            'NÃO confundir com ConvergenceTracker.converged_round (índice '
            'interno da janela deslizante, não o número da rodada — ver '
            'docstring desta migration). Comparar com best_round: quando os '
            'dois divergem bastante, é sinal de que "convergência" (métrica '
            'estável) não coincidiu com "melhor qualidade" (achado real no '
            'treino 85, DP-uniforme: best_round=4, convergence_round=74).';
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE metrics.fl_trainings DROP COLUMN IF EXISTS convergence_round;
    """)
