"""
convergence.py — Rastreamento de convergência do treinamento federado.

Algoritmo: janela deslizante sobre os últimos `patience` deltas consecutivos.
Um round ruidoso "envelhece" e sai da janela; não reinicia a contagem.
Isso é adequado para FL com dados hospitalares non-IID, onde rounds isolados
podem ser instáveis por natureza.

Propriedade de idempotência: uma vez convergido, check() sempre retorna True.
Isso garante consistência entre o valor booleano retornado e converged_round.

Usado pelo Caminho B (produção real, ver infrastructure/mosaicfl_server/
strategy/core.py). Correção 2026-07-28: o docstring afirmava "importado por
todos os adapters... nenhum redefine esta lógica localmente" — isso está
errado desde sempre. O Caminho A (experiments/training/core/fl_core/
manual_loop.py:278-313) reimplementa a lógica de convergência inline (delta
entre rodadas consecutivas + FED_CFG.min_rounds de warm-up, ambos ausentes
aqui), não importa esta classe. Os dois algoritmos são parecidos mas não
idênticos — não assumir paridade automática entre os dois caminhos."""
from __future__ import annotations


class ConvergenceTracker:
    """
    Detecta estabilização da accuracy global via janela deslizante.

    Converge quando todos os `patience` deltas dentro da janela são
    menores que `threshold`. Uma vez convergido, não reverte.

    Sem defaults propositalmente: threshold e patience são responsabilidade
    do adapter (experimento ou produção) que instancia este objeto.
    """

    def __init__(self, threshold: float, patience: int) -> None:
        self.threshold = threshold
        self.patience = patience
        self.history: list[float] = []
        self.converged_round: int | None = None

    def check(self, accuracy: float) -> bool:
        """
        Registra accuracy do round atual e verifica convergência.

        Returns:
            True se convergência foi atingida (inclusive em rounds anteriores).
            False enquanto histórico insuficiente ou janela ainda instável.
        """
        if self.converged_round is not None:
            self.history.append(accuracy)
            return True

        self.history.append(accuracy)

        if len(self.history) < self.patience + 1:
            return False

        recent = self.history[-(self.patience + 1):]
        deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]

        if all(d < self.threshold for d in deltas):
            self.converged_round = len(self.history)
            return True

        return False

    def reset(self) -> None:
        """Reinicia o rastreamento do zero."""
        self.history.clear()
        self.converged_round = None
