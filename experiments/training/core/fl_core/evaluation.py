"""evaluation.py — Avaliação do modelo global no conjunto de teste, por rodada."""
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from mosaicfl.core.config import DEVICE
from mosaicfl.core.model import SimplifiedBEHRT


def evaluate_global_model(
    model: SimplifiedBEHRT, test_loader: DataLoader
) -> Tuple[float, float, float, List[float]]:
    """Avalia modelo global no conjunto de teste.

    Retorna (loss, accuracy, f1_macro, per_class_f1).
    f1_macro é o critério primário de seleção do checkpoint — mais robusto que
    accuracy em datasets desbalanceados (zero_division=0 penaliza classes não previstas).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y, batch_dia in test_loader:
            batch_x   = batch_x.to(DEVICE)
            batch_y   = batch_y.to(DEVICE)
            batch_dia = batch_dia.to(DEVICE)
            logits    = model(batch_x, dia_relativo=batch_dia)
            loss      = criterion(logits, batch_y)
            loss_sum += loss.item() * batch_y.size(0)
            _, predicted = torch.max(logits, dim=1)
            total   += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    avg_loss     = loss_sum / total if total > 0 else 0.0
    accuracy     = correct  / total if total > 0 else 0.0
    f1_macro     = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    return avg_loss, accuracy, f1_macro, per_class_f1
