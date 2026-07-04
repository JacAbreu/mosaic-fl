#!/usr/bin/env python3
"""
run_bootstrap_ci.py — Intervalos de confiança via bootstrap sobre o melhor checkpoint.

Carrega o melhor modelo salvo no banco, coleta predições no conjunto de teste e aplica
bootstrap (reamostragem com reposição, n=1000) para estimar IC 95% de acurácia, F1 macro,
AUC macro e ECE. Persiste resultados em metrics.bootstrap_ci (PostgreSQL ou SQLite).

Uso:
    python experiments/training_runner/run_bootstrap_ci.py
    # ou via Makefile:
    make bootstrap-ci
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.makedirs("experiments/logs", exist_ok=True)

log_file = f"experiments/logs/bootstrap_ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from infrastructure.shared.checkpoint_store import get_checkpoint_store
from experiments.training.core.dataloaders import prepare_dataloaders_from_db
from mosaicfl.core.config import DEVICE, FL_DB_URL, MODEL_CFG

from mosaicfl.core.model import SimplifiedBEHRT

_CREATE_BOOTSTRAP_SQLITE = """
CREATE TABLE IF NOT EXISTS bootstrap_ci (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_round INTEGER NOT NULL,
    n_test           INTEGER NOT NULL,
    n_bootstrap      INTEGER NOT NULL,
    confidence       REAL    NOT NULL,
    metric           TEXT    NOT NULL,
    point_estimate   REAL    NOT NULL,
    mean             REAL    NOT NULL,
    std              REAL    NOT NULL,
    ci_low           REAL    NOT NULL,
    ci_high          REAL    NOT NULL,
    created_at       TEXT    NOT NULL
)
"""

_CREATE_BOOTSTRAP_PG = """
CREATE TABLE IF NOT EXISTS metrics.bootstrap_ci (
    id               SERIAL PRIMARY KEY,
    checkpoint_round INTEGER     NOT NULL,
    n_test           INTEGER     NOT NULL,
    n_bootstrap      INTEGER     NOT NULL,
    confidence       REAL        NOT NULL,
    metric           TEXT        NOT NULL,
    point_estimate   REAL        NOT NULL,
    mean             REAL        NOT NULL,
    std              REAL        NOT NULL,
    ci_low           REAL        NOT NULL,
    ci_high          REAL        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _save_to_db(db_url, checkpoint_round, n_test, n_bootstrap, confidence, point_estimates, ci_results):
    rows = []
    for metric, vals in ci_results.items():
        rows.append({
            "checkpoint_round": checkpoint_round,
            "n_test":           n_test,
            "n_bootstrap":      n_bootstrap,
            "confidence":       confidence,
            "metric":           metric,
            "point_estimate":   point_estimates.get(metric, 0.0),
            "mean":             vals["mean"],
            "std":              vals["std"],
            "ci_low":           vals["ci_low"],
            "ci_high":          vals["ci_high"],
        })

    if db_url:
        import sqlalchemy as sa
        engine = sa.create_engine(db_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(sa.text(_CREATE_BOOTSTRAP_PG))
            for row in rows:
                conn.execute(sa.text("""
                    INSERT INTO metrics.bootstrap_ci
                        (checkpoint_round, n_test, n_bootstrap, confidence, metric,
                         point_estimate, mean, std, ci_low, ci_high)
                    VALUES
                        (:checkpoint_round, :n_test, :n_bootstrap, :confidence, :metric,
                         :point_estimate, :mean, :std, :ci_low, :ci_high)
                """), row)
        logger.info(f"  Salvo em metrics.bootstrap_ci (PostgreSQL) — {len(rows)} linhas")
    else:
        import sqlite3
        db_path = "checkpoints/bootstrap_ci.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(_CREATE_BOOTSTRAP_SQLITE)
            created_at = datetime.utcnow().isoformat()
            for row in rows:
                conn.execute(
                    "INSERT INTO bootstrap_ci "
                    "(checkpoint_round, n_test, n_bootstrap, confidence, metric, "
                    " point_estimate, mean, std, ci_low, ci_high, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (row["checkpoint_round"], row["n_test"], row["n_bootstrap"],
                     row["confidence"], row["metric"], row["point_estimate"],
                     row["mean"], row["std"], row["ci_low"], row["ci_high"], created_at),
                )
        logger.info(f"  Salvo em {db_path} (SQLite) — {len(rows)} linhas")


def collect_predictions(model, loader):
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_dia in loader:
            batch_x   = batch_x.to(DEVICE)
            batch_y   = batch_y.to(DEVICE)
            batch_dia = batch_dia.to(DEVICE)
            logits    = model(batch_x, dia_relativo=batch_dia)
            proba     = torch.softmax(logits, dim=1)
            pred      = torch.argmax(logits, dim=1)
            all_true.extend(batch_y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_proba.extend(proba.cpu().tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_proba)


def _ece(y_true, y_proba, n_bins=10):
    confidence = np.max(y_proba, axis=1)
    predicted  = np.argmax(y_proba, axis=1)
    correct    = (predicted == y_true).astype(float)
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def bootstrap_ci(y_true, y_pred, y_proba, n_bootstrap=1000, confidence=0.95):
    rng = np.random.default_rng(42)
    n   = len(y_true)
    acc_s, f1_s, auc_s, ece_s = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        acc_s.append(accuracy_score(yt, yp))
        f1_s.append(f1_score(yt, yp, average="macro", zero_division=0))
        try:
            auc_s.append(roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
        except ValueError:
            pass
        ece_s.append(_ece(yt, ypr))

    alpha = (1.0 - confidence) / 2

    def _stat(values):
        arr = np.array(values)
        return {
            "mean":     round(float(np.mean(arr)), 4),
            "std":      round(float(np.std(arr)),  4),
            "ci_low":   round(float(np.percentile(arr, alpha * 100)),        4),
            "ci_high":  round(float(np.percentile(arr, (1 - alpha) * 100)), 4),
        }

    return {
        "accuracy":  _stat(acc_s),
        "f1_macro":  _stat(f1_s),
        "auc_macro": _stat(auc_s),
        "ece":       _stat(ece_s),
    }


def main():
    logger.info("=" * 60)
    logger.info("BOOTSTRAP CI — Intervalos de Confiança 95%")
    logger.info("=" * 60)

    if not FL_DB_URL:
        logger.error("FL_DB_URL não configurado.")
        sys.exit(1)

    logger.info("[1/4] Carregando dados do banco...")
    _, test_loader, _, _, _, _, _, _, _ = prepare_dataloaders_from_db(FL_DB_URL)
    n_test = len(test_loader.dataset)
    logger.info(f"  Conjunto de teste: {n_test} amostras")

    logger.info("[2/4] Carregando melhor checkpoint...")
    store = get_checkpoint_store(FL_DB_URL)
    ckpt  = store.load_best()
    if ckpt is None:
        logger.error("Nenhum checkpoint encontrado.")
        sys.exit(1)
    best_round = ckpt.get("checkpoint_round", 0)
    logger.info(f"  Checkpoint: rodada {best_round}")

    model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    logger.info("[3/4] Coletando predições...")
    y_true, y_pred, y_proba = collect_predictions(model, test_loader)
    point = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro":  round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "auc_macro": round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4),
        "ece":       round(_ece(y_true, y_proba), 4),
    }
    logger.info(f"  Estimativas pontuais: {point}")

    logger.info("[4/4] Bootstrap (n=1.000 iterações, IC 95%)...")
    ci = bootstrap_ci(y_true, y_pred, y_proba)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS — IC 95%")
    logger.info("=" * 60)
    for metric, vals in ci.items():
        logger.info(
            f"  {metric:<12} ponto={point[metric]:.4f}  "
            f"IC=[{vals['ci_low']:.4f} – {vals['ci_high']:.4f}]  std={vals['std']:.4f}"
        )

    logger.info("[5/4] Persistindo no banco...")
    _save_to_db(FL_DB_URL, best_round, n_test, 1000, 0.95, point, ci)

    # Resumo compacto em JSON (sem vetores — apenas métricas agregadas)
    summary = {
        "checkpoint_round": best_round,
        "n_test": n_test,
        "point_estimates": point,
        "bootstrap_ci_95pct": {m: {"mean": v["mean"], "ci": [v["ci_low"], v["ci_high"]]} for m, v in ci.items()},
    }
    out_path = f"experiments/logs/bootstrap_ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Resumo: {out_path} | Log: {log_file}")


if __name__ == "__main__":
    main()
