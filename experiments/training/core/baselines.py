"""Baseline comparativo: Random Forest (Bag-of-Tokens) centralizado e por hospital."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mosaicfl.core.config import MODEL_CFG, RANDOM_SEED, VOCAB_SIZE

logger = logging.getLogger(__name__)


def _loader_to_bow(loader: DataLoader, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Converte sequências de token IDs de um DataLoader em vetores Bag-of-Tokens."""
    X, y = [], []
    for batch_x, batch_y, *_ in loader:
        for seq, label in zip(batch_x.numpy(), batch_y.numpy()):
            bow = np.zeros(vocab_size, dtype=np.float32)
            for tok in seq:
                if 2 < int(tok) < vocab_size:
                    bow[int(tok)] += 1.0
            X.append(bow)
            y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _safe_float(v: Optional[float]) -> Optional[float]:
    """Converte NaN/inf para None — garante JSON válido."""
    import math
    if v is None:
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)


def _eval_rf(
    rf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: List[str],
) -> Dict:
    """Avalia um RandomForestClassifier e retorna métricas no mesmo formato do EvaluationReport."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from mosaicfl.core.evaluation import compute_ece

    y_prob = rf.predict_proba(X_test)
    y_pred = rf.predict(X_test)

    n_classes = len(class_labels)
    rf_classes = list(rf.classes_)
    y_prob_ordered = np.zeros((len(y_test), n_classes), dtype=np.float32)
    for j, cls in enumerate(rf_classes):
        if cls < n_classes:
            y_prob_ordered[:, cls] = y_prob[:, j]

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    try:
        macro_auc: Optional[float] = float(
            roc_auc_score(y_test, y_prob_ordered, multi_class="ovr", average="macro")
        )
    except (ValueError, TypeError):
        macro_auc = None

    confidences = torch.tensor(y_prob_ordered.max(axis=1), dtype=torch.float32)
    correct     = torch.tensor(y_pred == y_test, dtype=torch.bool)
    cal         = compute_ece(confidences, correct)

    per_class_auc: Dict[str, Optional[float]] = {}
    for i, label in enumerate(class_labels):
        y_bin = (y_test == i).astype(int)
        try:
            auc: Optional[float] = float(roc_auc_score(y_bin, y_prob_ordered[:, i]))
        except (ValueError, TypeError):
            auc = None
        per_class_auc[label] = _safe_float(auc)

    return {
        "accuracy":      round(accuracy, 4),
        "macro_f1":      round(macro_f1, 4),
        "macro_auc":     _safe_float(macro_auc),
        "ece":           round(cal.ece, 4),
        "per_class_auc": per_class_auc,
    }


def run_baseline_rf(
    client_loaders: Dict,
    test_loader: DataLoader,
    class_labels: List[str] = None,
    vocab_size: int = None,
    random_seed: int = None,
) -> Dict:
    """
    Baseline Random Forest (Bag-of-Tokens) para comparação com SimplifiedBEHRT.

    Opção A — RF Centralizado: pool de todos os dados de treino dos clientes.
    Opção B — RF por Hospital: um RF independente por cliente.

    A diferença BEHRT(FL) − RF(hospital) mede o ganho do aprendizado federado.
    A diferença BEHRT(FL) − RF(centralizado) mede o ganho da modelagem sequencial.
    """
    from sklearn.ensemble import RandomForestClassifier

    class_labels = list(class_labels or MODEL_CFG.class_labels)
    vocab_size   = vocab_size  if vocab_size  is not None else VOCAB_SIZE
    random_seed  = random_seed if random_seed is not None else RANDOM_SEED

    logger.info("=" * 60)
    logger.info("BASELINE — Random Forest (Bag-of-Tokens)")
    logger.info("=" * 60)
    logger.info(f"  vocab_size: {vocab_size} | classes: {class_labels}")

    X_test, y_test = _loader_to_bow(test_loader, vocab_size)
    unique, counts = np.unique(y_test, return_counts=True)
    dist = {class_labels[int(c)]: int(n) for c, n in zip(unique, counts) if int(c) < len(class_labels)}
    logger.info(f"  Teste: {len(X_test)} amostras | distribuição: {dist}")

    results: Dict = {}

    logger.info("\n[Opção A] RF Centralizado — pool de todos os clientes")
    X_parts, y_parts = [], []
    for cid, (train_loader, _) in client_loaders.items():
        X_c, y_c = _loader_to_bow(train_loader, vocab_size)
        X_parts.append(X_c)
        y_parts.append(y_c)
        logger.info(f"  Cliente {cid}: {len(X_c)} amostras")

    X_pool = np.concatenate(X_parts, axis=0)
    y_pool = np.concatenate(y_parts, axis=0)
    logger.info(f"  Pool total: {len(X_pool)} amostras")

    rf_central = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=random_seed, n_jobs=-1,
    )
    rf_central.fit(X_pool, y_pool)
    m_a = _eval_rf(rf_central, X_test, y_test, class_labels)
    results["opcao_a_centralizado"] = {
        **m_a,
        "n_train": int(len(X_pool)),
        "descricao": "RF treinado em pool de todos os clientes (centralizado)",
    }
    auc_str = f"{m_a['macro_auc']:.4f}" if m_a['macro_auc'] is not None else "n/a"
    logger.info(
        f"  Resultado → Acc={m_a['accuracy']:.4f}  "
        f"AUC={auc_str}  F1={m_a['macro_f1']:.4f}  ECE={m_a['ece']:.4f}"
    )

    logger.info("\n[Opção B] RF por Hospital — modelo independente por cliente")
    per_client: Dict = {}
    for cid, (train_loader, _) in client_loaders.items():
        X_c, y_c = _loader_to_bow(train_loader, vocab_size)
        n_local_classes = int(len(np.unique(y_c)))
        rf_local = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_seed, n_jobs=-1,
        )
        rf_local.fit(X_c, y_c)
        m = _eval_rf(rf_local, X_test, y_test, class_labels)
        per_client[str(cid)] = {
            **m,
            "n_train": int(len(X_c)),
            "n_classes_local": n_local_classes,
        }
        auc_h = f"{m['macro_auc']:.4f}" if m['macro_auc'] is not None else "n/a"
        logger.info(
            f"  Hospital {cid}: Acc={m['accuracy']:.4f}  AUC={auc_h}  "
            f"F1={m['macro_f1']:.4f}  (treino={len(X_c)}, classes_locais={n_local_classes})"
        )
    results["opcao_b_por_hospital"] = {
        "per_client": per_client,
        "descricao": "RF independente por hospital — sem compartilhamento de dados",
    }

    def _fmt(v: Optional[float]) -> str:
        return f"{v:>8.4f}" if v is not None else "     n/a"

    logger.info("\n" + "=" * 60)
    logger.info("TABELA COMPARATIVA — Baseline vs. BEHRT (FL)")
    logger.info("=" * 60)
    logger.info(f"{'Modelo':<42} {'Accuracy':>8} {'AUC-ROC':>8} {'F1 Macro':>8} {'ECE':>7}")
    logger.info("-" * 76)
    logger.info(
        f"{'RF Centralizado (BoT)':<42} "
        f"{_fmt(m_a['accuracy'])} {_fmt(m_a['macro_auc'])} "
        f"{_fmt(m_a['macro_f1'])} {_fmt(m_a['ece'])}"
    )
    for cid, m in per_client.items():
        logger.info(
            f"  {'RF Hospital ' + str(cid) + ' (BoT)':<40} "
            f"{_fmt(m['accuracy'])} {_fmt(m['macro_auc'])} "
            f"{_fmt(m['macro_f1'])} {_fmt(m['ece'])}"
        )
    logger.info("-" * 76)
    logger.info(f"{'SimplifiedBEHRT (FL) — ver evaluation_round_*.json':<42}")
    logger.info("=" * 60)

    results["meta"] = {
        "vocab_size":   vocab_size,
        "n_test":       int(len(X_test)),
        "class_labels": class_labels,
        "rf_params":    {"n_estimators": 200, "class_weight": "balanced"},
    }
    return results
