"""errors.py — Exceção de falha total da cadeia de fallback de carregamento."""
from typing import Dict, List


class DataLoadError(Exception):
    """
    Levantada quando todas as fontes de dados falharam em load_with_fallback().
    Inclui o histórico de tentativas para facilitar o diagnóstico.
    """
    def __init__(self, message: str, attempts: List[Dict] = None):
        self.attempts = attempts or []
        super().__init__(message)

    def __str__(self):
        base = super().__str__()
        if self.attempts:
            log = "\n".join(
                f"  [{i+1}] {a['fonte']}: {a['erro']}"
                for i, a in enumerate(self.attempts)
            )
            return f"{base}\n\nTentativas realizadas:\n{log}"
        return base
