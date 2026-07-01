"""base.py — Interface abstrata para persistência de métricas de avaliação federada."""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class MetricsStore(ABC):
    """Interface para persistência de métricas de avaliação federada."""

    @abstractmethod
    def save(
        self,
        round_num: int,
        metrics: Dict,
        checkpoint_sha256: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        """
        Persiste métricas de avaliação de um round.

        metrics pode conter:
          accuracy, loss, macro_auc, macro_f1, ece           — métricas globais do modelo
          per_class_auc, per_class_f1                        — dicts {classe: valor}
          rag_precision_at_k, rag_k, rag_per_class_precision — métricas do RAG
        """

    @abstractmethod
    def load_history(self, last_n: Optional[int] = None) -> list:
        """Retorna histórico de métricas, opcionalmente limitado aos últimos N rounds."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna as métricas do round mais recente, ou None."""
