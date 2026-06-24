"""
outcome_evaluator.py
Avaliação de qualidade do modelo em produção com ground truth tardio.

Fluxo:
  1. /api/exams/ingest → predição salva em predicted_outcomes (com correlation_token)
  2. Alta do paciente  → desfecho registrado em outcome_feedback via /api/patients/{id}/outcome
  3. evaluate_production_outcomes() → lê pares, computa métricas, salva no MetricsStore

Por que ground truth tardio?
  O desfecho real (alta, óbito, internação prolongada) só é conhecido no momento da alta —
  dias ou semanas após a predição. O correlation_token efêmero é o único elo entre predição
  e desfecho: nunca trafega pela rede FL e não expõe identidade do paciente.
"""
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MIN_PAIRS_DEFAULT = 30


def evaluate_production_outcomes(
    db,
    metrics_store,
    min_pairs: int = _MIN_PAIRS_DEFAULT,
) -> Optional[Dict]:
    """
    Computa métricas de produção a partir dos pares (predição, desfecho real).

    Retorna dict com as métricas ou None se não houver pares suficientes.
    Salva automaticamente no MetricsStore com data_source='production'.

    Requer min_pairs para garantir estimativas estatisticamente estáveis
    (AUC com < 30 amostras tem intervalos de confiança muito amplos).
    """
    pairs = db.get_prediction_outcome_pairs()
    n = len(pairs)

    if n < min_pairs:
        logger.info(
            "outcome_evaluator_skipped pairs=%d min_required=%d",
            n, min_pairs,
        )
        return None

    y_true_labels = [p["actual_label"] for p in pairs]
    y_pred_labels = [p["predicted_label"] for p in pairs]

    # Constrói mapeamento label→class a partir dos pares de predição.
    # predicted_class vem do modelo; actual_class pode ser None (não informado pelo caller).
    label_to_class: Dict[str, int] = {}
    for p in pairs:
        label_to_class[p["predicted_label"]] = int(p["predicted_class"])
        if p.get("actual_class") is not None:
            label_to_class[p["actual_label"]] = int(p["actual_class"])

    y_true_int = [label_to_class.get(lbl) for lbl in y_true_labels]
    y_pred_int = [label_to_class.get(lbl) for lbl in y_pred_labels]

    # Filtra pares onde actual_label não está no vocabulário do modelo
    valid = [(yt, yp, pair) for yt, yp, pair in zip(y_true_int, y_pred_int, pairs)
             if yt is not None and yp is not None]

    if not valid:
        logger.warning("outcome_evaluator: nenhum par com label reconhecido — verifique o vocabulário")
        return None

    yt_valid = [v[0] for v in valid]
    yp_valid = [v[1] for v in valid]
    pairs_valid = [v[2] for v in valid]

    accuracy = sum(1 for a, b in zip(yt_valid, yp_valid) if a == b) / len(yt_valid)

    metrics: Dict = {
        "accuracy": round(accuracy, 4),
        "n_pairs":  len(valid),
        "n_total":  n,
    }

    try:
        from sklearn.metrics import f1_score, roc_auc_score
        import numpy as np

        classes = sorted(label_to_class.values())
        n_classes = len(classes)

        macro_f1 = f1_score(yt_valid, yp_valid, average="macro", zero_division=0)
        metrics["macro_f1"] = round(float(macro_f1), 4)

        per_class_f1_arr = f1_score(yt_valid, yp_valid, average=None, zero_division=0, labels=classes)
        class_id_to_label = {v: k for k, v in label_to_class.items()}
        metrics["per_class_f1"] = {
            class_id_to_label.get(c, str(c)): round(float(f), 4)
            for c, f in zip(classes, per_class_f1_arr)
        }

        # AUC — requer probabilidades calibradas. Extraídas de class_probabilities (JSON).
        proba_matrix = _build_proba_matrix(pairs_valid, classes, label_to_class)
        if proba_matrix is not None and n_classes >= 2:
            try:
                if n_classes == 2:
                    auc = roc_auc_score(yt_valid, proba_matrix[:, 1])
                else:
                    auc = roc_auc_score(
                        yt_valid, proba_matrix,
                        multi_class="ovr", average="macro",
                        labels=classes,
                    )
                metrics["macro_auc"] = round(float(auc), 4)
            except ValueError as exc:
                logger.warning("outcome_evaluator: AUC não computável — %s", exc)

        # ECE — calibração de probabilidades (binning em 10 bins)
        ece = _compute_ece(yt_valid, yp_valid, proba_matrix, classes)
        if ece is not None:
            metrics["ece"] = round(ece, 4)

    except ImportError:
        logger.warning("outcome_evaluator: sklearn não disponível — métricas limitadas a accuracy")

    metrics_store.save(
        round_num=-1,
        metrics=metrics,
        data_source="production",
    )

    logger.info(
        "outcome_evaluated pairs=%d accuracy=%.4f f1=%s auc=%s ece=%s",
        len(valid),
        accuracy,
        f"{metrics['macro_f1']:.4f}" if "macro_f1" in metrics else "n/a",
        f"{metrics['macro_auc']:.4f}" if "macro_auc" in metrics else "n/a",
        f"{metrics['ece']:.4f}"       if "ece"       in metrics else "n/a",
    )
    return metrics


def _build_proba_matrix(pairs: List[dict], classes: List[int], label_to_class: Dict[str, int]):
    """Constrói matriz [n_samples, n_classes] de probabilidades a partir dos pares."""
    try:
        import numpy as np
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n = len(pairs)
        matrix = np.zeros((n, len(classes)), dtype=float)
        for i, pair in enumerate(pairs):
            proba_dict = json.loads(pair["class_probabilities"])
            for label, vals in proba_dict.items():
                cls_id = label_to_class.get(label)
                if cls_id is not None and cls_id in class_to_idx:
                    matrix[i, class_to_idx[cls_id]] = float(vals.get("value", 0.0))
        return matrix
    except Exception as exc:
        logger.warning("outcome_evaluator: erro ao construir matriz de probabilidades — %s", exc)
        return None


def _compute_ece(y_true, y_pred, proba_matrix, classes, n_bins: int = 10) -> Optional[float]:
    """Expected Calibration Error por binning em n_bins intervalos."""
    if proba_matrix is None:
        return None
    try:
        import numpy as np
        class_to_idx = {c: i for i, c in enumerate(classes)}
        confidences = np.array([
            proba_matrix[i, class_to_idx[yp]]
            for i, yp in enumerate(y_pred)
            if yp in class_to_idx
        ])
        correctness = np.array([
            float(yt == yp)
            for yt, yp in zip(y_true, y_pred)
            if yp in class_to_idx
        ])
        if len(confidences) == 0:
            return None
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(confidences)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_acc  = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)
    except Exception as exc:
        logger.warning("outcome_evaluator: ECE não computável — %s", exc)
        return None
