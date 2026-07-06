"""base.py — Interface abstrata para fontes de dados do cliente federado + constantes padrão."""
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch.utils.data import DataLoader

# ── Constantes ─────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = int(os.getenv("FL_BATCH_SIZE", "16"))
DEFAULT_SEQ_LEN = int(os.getenv("FL_SEQ_LEN", "128"))
DEFAULT_VOCAB_SIZE = int(os.getenv("FL_VOCAB_SIZE", "10000"))
DEFAULT_NUM_SAMPLES = int(os.getenv("FL_SIM_SAMPLES", "200"))


class DataSource(ABC):
    """Interface para todas as fontes de dados do cliente."""

    @abstractmethod
    def load(self, vocab: Optional[dict] = None) -> DataLoader:
        """Retorna DataLoader PyTorch pronto para treinamento.

        vocab: vocabulário canônico compartilhado (enviado pelo servidor via config
        da rodada, em produção). Fontes que não usam vocab (simulated, csv) ignoram.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """Retorna metadados sobre a fonte (nome, tipo, registros, etc.)."""
        pass

    def validate(self) -> Tuple[bool, str]:
        """Valida se a fonte está acessível. Retorna (ok, mensagem)."""
        return True, "Validação padrão: OK"
