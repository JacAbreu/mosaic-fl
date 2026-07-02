"""router.py — Roteia entre o loop manual e o loop Ray, baseado em config.USE_RAY."""
import logging
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from mosaicfl.core.config import USE_RAY
from mosaicfl.core.model import SimplifiedBEHRT

from .manual_loop import run_federated_learning_manual
from .ray_loop import run_federated_learning_ray

logger = logging.getLogger(__name__)


def run_federated_learning(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
    cal_loader: DataLoader = None,
    test_loader_origin: Optional[DataLoader] = None,
    origin_labels: Optional[Dict[int, str]] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """
    Roteia para o modo correto baseado em config.USE_RAY.

    Para alternar:
        Edite mosaicfl/core/config.py → USE_RAY = False  (manual, leve)
        Edite mosaicfl/core/config.py → USE_RAY = True   (Ray, paralelo)

    test_loader_origin/origin_labels: avaliação por subgrupo de origem hospitalar
    no checkpoint final (Experimento 3 — contraste non-IID vs. iid_simulado).
    Só suportado no loop manual — o loop Ray não participa do tracking de
    training_id/fl_trainings, então não há onde persistir o resultado.
    """
    if bool(USE_RAY):
        logger.info("Modo Ray ativado (USE_RAY=True).")
        return run_federated_learning_ray(client_loaders, test_loader, total_train_samples, vocab=vocab)
    else:
        logger.info("Modo manual ativado (USE_RAY=False). Ray NÃO é necessário.")
        return run_federated_learning_manual(
            client_loaders, test_loader, total_train_samples, vocab=vocab, cal_loader=cal_loader,
            test_loader_origin=test_loader_origin, origin_labels=origin_labels,
        )
