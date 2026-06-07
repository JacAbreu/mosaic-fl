"""
federated.py — Utilitários de agregação federada.

Funções de agregação ponderada usadas por ambos os adapters
(produção e experimento) para consolidar métricas dos clientes.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from collections import OrderedDict


def _weighted_average(metrics: List[Tuple[int, Dict]], key: str) -> Dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {key: 0.0}
    return {key: sum(n * m.get(key, 0.0) for n, m in metrics) / total}


def weighted_average_accuracy(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Média ponderada de accuracy — para evaluate_metrics_aggregation_fn."""
    return _weighted_average(metrics, "accuracy")


def weighted_average_loss(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Média ponderada de loss — para fit_metrics_aggregation_fn."""
    return _weighted_average(metrics, "loss")


# Alias mantido para compatibilidade com código existente
weighted_average = weighted_average_accuracy


def get_evaluate_fn(test_loader) -> Callable:
    """
    Retorna a função de avaliação global usada pelo servidor a cada rodada.

    Compatível com evaluate_fn do Flower — avalia o modelo global no
    test_loader fornecido e retorna (loss, {"accuracy": float}).
    """
    from mosaicfl.core.model_v2 import SimplifiedBEHRT
    from mosaicfl.core.config import RUNTIME_CFG

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict,
    ) -> Tuple[float, Dict]:
        model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(RUNTIME_CFG.device), batch_y.to(RUNTIME_CFG.device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                _, predicted = torch.max(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"  [Servidor] Rodada {server_round:>3} | "
            f"Loss global: {avg_loss:.4f} | Acurácia global: {accuracy:.4f}"
        )
        return avg_loss, {"accuracy": accuracy, "rodada": server_round}

    return evaluate
