"""
evaluation.py — Métricas de avaliação clínica para o MOSAIC-FL.

Além de accuracy, modelos usados em decisão clínica devem reportar:
  - ECE  (Expected Calibration Error): o quanto as probabilidades emitidas
          correspondem às frequências reais de acerto. Um modelo que diz "80%
          de confiança" deve acertar em ~80% dos casos com essa confiança.
  - AUC-ROC por classe (one-vs-rest): detecta se o modelo é incapaz de
          distinguir uma classe específica, mesmo com accuracy global alta.
  - F1 por classe e macro: penaliza tanto falsos positivos quanto negativos.
  - Matriz de confusão: onde o modelo erra e para qual classe migra o erro.
  - Diagrama de confiabilidade: visualização binned de confiança vs. acurácia.

Referências:
  Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.
  Naeini et al. "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------

@dataclass
class BinStats:
    """Estatísticas de um bin do diagrama de confiabilidade."""
    confidence_mean: float   # confiança média do bin
    accuracy:        float   # acurácia real no bin
    count:           int     # amostras no bin
    gap:             float   # |confidence_mean - accuracy|


@dataclass
class CalibrationResult:
    """
    Resultado completo da avaliação de calibração.

    ece:         Expected Calibration Error ∈ [0, 1]. Abaixo de 0.05 é bom.
                 Acima de 0.10 é problemático para uso clínico.
    mce:         Maximum Calibration Error — pior bin. Relevante para detectar
                 regiões de confiança sistematicamente erradas.
    bins:        Estatísticas por bin para o diagrama de confiabilidade.
    temperature: T usado (1.0 = sem calibração).
    """
    ece:         float
    mce:         float
    bins:        List[BinStats]
    temperature: float = 1.0
    n_samples:   int   = 0


@dataclass
class ClassMetrics:
    """Métricas por classe (one-vs-rest)."""
    auc_roc:   float
    f1:        float
    precision: float
    recall:    float
    support:   int


@dataclass
class EvaluationReport:
    """Relatório completo de avaliação clínica."""
    accuracy:          float
    macro_f1:          float
    macro_auc:         float
    per_class:         Dict[str, ClassMetrics]
    calibration:       CalibrationResult
    confusion_matrix:  List[List[int]]
    n_samples:         int
    class_labels:      List[str]


# ---------------------------------------------------------------------------
# ECE e diagrama de confiabilidade
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: torch.Tensor,
    correct:     torch.Tensor,
    n_bins:      int = 15,
) -> CalibrationResult:
    """
    Calcula ECE e estatísticas por bin para o diagrama de confiabilidade.

    Args:
        confidences: (N,) — probabilidade máxima emitida para cada amostra.
        correct:     (N,) bool — se a classe predita era a correta.
        n_bins:      número de bins uniformes em [0, 1].

    Returns:
        CalibrationResult com ECE, MCE e lista de BinStats.

    Fórmula:
        ECE = Σ_b (|B_b| / N) * |acc(B_b) − conf(B_b)|
    """
    n = len(confidences)
    bins: List[BinStats] = []
    weighted_gaps: List[float] = []

    edges = torch.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(mask.sum())
        if count == 0:
            continue

        conf_mean = float(confidences[mask].mean())
        acc       = float(correct[mask].float().mean())
        gap       = abs(conf_mean - acc)

        bins.append(BinStats(
            confidence_mean=round(conf_mean, 4),
            accuracy=round(acc, 4),
            count=count,
            gap=round(gap, 4),
        ))
        weighted_gaps.append((count / n) * gap)

    ece = float(sum(weighted_gaps))
    mce = float(max((b.gap for b in bins), default=0.0))

    return CalibrationResult(ece=round(ece, 4), mce=round(mce, 4), bins=bins, n_samples=n)


# ---------------------------------------------------------------------------
# Coleta de logits e labels de um DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(
    model:       torch.nn.Module,
    loader:      DataLoader,
    device:      str = "cpu",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Passa o DataLoader pelo modelo e coleta (logits calibrados, labels).

    Args:
        temperature: divide os logits (T > 1 suaviza, T < 1 afila). Use 1.0
                     para medir calibração *antes* de calibrar, e o T ajustado
                     para medir *depois*.

    Returns:
        all_probs:  (N, num_classes) — softmax(logits / T)
        all_labels: (N,)             — labels reais
    """
    model.eval()
    model.to(device)
    probs_list:  List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for batch in loader:
        batch_x = batch[0].to(device)
        batch_y = batch[1]
        batch_dia = batch[2].to(device) if len(batch) > 2 else None
        mask = (batch_x == 0)
        logits = model(batch_x, mask=mask, dia_relativo=batch_dia)
        probs  = F.softmax(logits / max(temperature, 1e-3), dim=-1).cpu()
        probs_list.append(probs)
        labels_list.append(batch_y)

    return torch.cat(probs_list), torch.cat(labels_list)


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------

def evaluate(
    model:        torch.nn.Module,
    loader:       DataLoader,
    class_labels: Sequence[str],
    device:       str = "cpu",
    temperature:  float = 1.0,
    n_bins:       int = 15,
) -> EvaluationReport:
    """
    Gera relatório completo de avaliação clínica.

    Inclui accuracy, F1 macro, AUC-ROC por classe (one-vs-rest),
    matriz de confusão e ECE/MCE para o diagrama de confiabilidade.

    Args:
        temperature: T do checkpoint. Passa 1.0 para ver calibração bruta,
                     passa o T salvo para ver após calibração.
    """
    try:
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score,
            confusion_matrix as sk_confusion_matrix,
        )
        import numpy as np
    except ImportError as e:
        raise ImportError("scikit-learn é necessário para avaliação clínica.") from e

    all_probs, all_labels = collect_logits(model, loader, device, temperature)

    n_classes = all_probs.shape[1]
    predicted = all_probs.argmax(dim=1)
    y_true    = all_labels.numpy()
    y_pred    = predicted.numpy()
    y_prob    = all_probs.numpy()

    accuracy  = float((predicted == all_labels).float().mean())
    macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC-ROC one-vs-rest (requer pelo menos 2 classes presentes por class)
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
        logger.warning("AUC-ROC não calculável — provavelmente só uma classe presente no loader.")

    # Por classe
    per_class: Dict[str, ClassMetrics] = {}
    for i, label in enumerate(class_labels):
        y_bin = (y_true == i).astype(int)
        try:
            auc = float(roc_auc_score(y_bin, y_prob[:, i]))
        except ValueError:
            auc = float("nan")
        support = int(y_bin.sum())
        per_class[label] = ClassMetrics(
            auc_roc   = round(auc, 4),
            f1        = round(float(f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)), 4),
            precision = round(float(precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)), 4),
            recall    = round(float(recall_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)), 4),
            support   = support,
        )

    # ECE com a confiança máxima (classe predita)
    confidences = all_probs.max(dim=1).values
    correct     = (predicted == all_labels)
    cal         = compute_ece(confidences, correct, n_bins=n_bins)
    cal.temperature = temperature

    # Matriz de confusão
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    return EvaluationReport(
        accuracy         = round(accuracy, 4),
        macro_f1         = round(macro_f1, 4),
        macro_auc        = round(macro_auc, 4),
        per_class        = per_class,
        calibration      = cal,
        confusion_matrix = cm,
        n_samples        = len(y_true),
        class_labels     = list(class_labels),
    )


# ---------------------------------------------------------------------------
# Impressão legível
# ---------------------------------------------------------------------------

def print_report(report: EvaluationReport) -> None:
    """Imprime o relatório de avaliação em formato legível."""
    cal = report.calibration
    print(f"\n{'='*60}")
    print(f"  RELATÓRIO DE AVALIAÇÃO CLÍNICA  (T={cal.temperature:.4f})")
    print(f"{'='*60}")
    print(f"  Amostras:    {report.n_samples}")
    print(f"  Accuracy:    {report.accuracy:.4f}  ({report.accuracy*100:.1f}%)")
    print(f"  Macro F1:    {report.macro_f1:.4f}")
    print(f"  Macro AUC:   {report.macro_auc:.4f}")
    print(f"\n  Calibração")
    print(f"  ├─ ECE:  {cal.ece:.4f}  {'✓ bom (<0.05)' if cal.ece < 0.05 else '⚠ moderado (<0.10)' if cal.ece < 0.10 else '✗ crítico (≥0.10)'}")
    print(f"  └─ MCE:  {cal.mce:.4f}  (pior bin)")

    print(f"\n  Por classe:")
    header = f"  {'Classe':<22} {'AUC':>6} {'F1':>6} {'Recall':>8} {'Prec':>6} {'N':>6}"
    print(header)
    print(f"  {'-'*56}")
    for lbl, m in report.per_class.items():
        auc_str = f"{m.auc_roc:.4f}" if not (m.auc_roc != m.auc_roc) else "  N/A "
        print(f"  {lbl:<22} {auc_str:>6} {m.f1:>6.4f} {m.recall:>8.4f} {m.precision:>6.4f} {m.support:>6}")

    print(f"\n  Matriz de confusão (linhas=real, colunas=predito):")
    short = [l[:8] for l in report.class_labels]
    header_cm = "           " + "".join(f"{s:>10}" for s in short)
    print(f"  {header_cm}")
    for i, row in enumerate(report.confusion_matrix):
        print(f"  {short[i]:<10}" + "".join(f"{v:>10}" for v in row))

    print(f"\n  Diagrama de confiabilidade ({len(cal.bins)} bins preenchidos):")
    print(f"  {'Conf':>8} {'Acc':>8} {'Gap':>8} {'N':>6}  Barra")
    print(f"  {'-'*50}")
    for b in cal.bins:
        bar_conf = "█" * int(b.confidence_mean * 20)
        bar_acc  = "░" * int(b.accuracy * 20)
        gap_flag = " ←OVERCONF" if b.confidence_mean > b.accuracy + 0.05 else (" ←UNDERCONF" if b.accuracy > b.confidence_mean + 0.05 else "")
        print(f"  {b.confidence_mean:>8.3f} {b.accuracy:>8.3f} {b.gap:>8.3f} {b.count:>6}  {gap_flag}")
    print()
