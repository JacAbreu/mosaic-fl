"""data_utils.py — Split treino/validação e cache de DataLoaders entre rounds."""
import logging
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from mosaicfl.core.config import FED_CFG

logger = logging.getLogger(__name__)


def parse_client_id(client_id: str) -> int:
    """Converte ID do hospital (string) para inteiro usado pelo FedProxClient."""
    try:
        return int(client_id)
    except ValueError:
        return abs(hash(client_id)) % 10_000


def _split_loader(loader: DataLoader, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Separa um DataLoader em treino e validação."""
    dataset = loader.dataset
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    if n_train < 1:
        return loader, loader
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(FED_CFG.random_seed),
    )
    return (
        DataLoader(train_ds, batch_size=loader.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=loader.batch_size, shuffle=False),
    )


# Cache de DataLoaders por (source_type, hospital_id) — evita recarregar o banco a cada round
_loader_cache: dict[tuple, Tuple[DataLoader, DataLoader]] = {}
