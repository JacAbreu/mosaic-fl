"""simulated.py — Fonte de dados sintéticos para TCC/prototipagem."""
import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import DEFAULT_BATCH_SIZE, DEFAULT_NUM_SAMPLES, DEFAULT_SEQ_LEN, DEFAULT_VOCAB_SIZE, DataSource

logger = logging.getLogger(__name__)


class SimulatedDataSource(DataSource):
    """
    Gera dados sintéticos para desenvolvimento e testes.
    Útil quando não há acesso ao SGBD real ou para benchmark.
    """

    def __init__(
        self,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_len: int = DEFAULT_SEQ_LEN,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        num_classes: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 42,
        hospital_id: Optional[str] = None,  # aceito mas ignorado — sem partição em modo sintético
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.seed = seed

    def load(self) -> DataLoader:
        logger.info(
            f"[SIMULATED] Gerando {self.num_samples} amostras sintéticas "
            f"(seq_len={self.seq_len}, vocab_size={self.vocab_size})"
        )
        torch.manual_seed(self.seed)

        X = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_len))
        y = torch.randint(0, self.num_classes, (self.num_samples,))

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"[SIMULATED] DataLoader pronto: {len(loader)} batches")
        return loader

    def get_metadata(self) -> dict:
        return {
            "type": "simulated",
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
        }

    def validate(self) -> Tuple[bool, str]:
        return True, "Fonte simulada: sempre disponível"
