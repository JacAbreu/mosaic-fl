#!/usr/bin/env python3
"""
run_seed_sensitivity.py — Análise de sensibilidade a seed: FedAvg vs FedNova.

Roda N rounds curtos (padrão: 30) para cada combinação (método × seed).
Checkpoints dos runs curtos vão para SQLite isolado — sem tocar no PostgreSQL de produção.
Resultados agregados (best_accuracy por run) persistem em metrics.sensitivity_runs.

O split de dados é FIXO (RANDOM_SEED=42 via randperm determinístico).
Apenas inicialização dos pesos e ordem dos batches variam por seed.

Uso:
    python experiments/run_seed_sensitivity.py [--rounds 30] [--seeds 42 7 123]
    # ou via Makefile:
    make seed-sensitivity
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

log_file = f"experiments/logs/seed_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from infrastructure.shared.checkpoint_store import SQLiteCheckpointStore
from experiments.training.dataloaders import prepare_dataloaders_from_db
from experiments.training.fl_core import run_federated_learning_manual
from mosaicfl.core.config import FL_DB_URL

_CREATE_SENSITIVITY_PG = """
CREATE TABLE IF NOT EXISTS metrics.sensitivity_runs (
    id             SERIAL PRIMARY KEY,
    experiment     TEXT        NOT NULL,
    method         TEXT        NOT NULL,
    seed           INTEGER     NOT NULL,
    n_rounds       INTEGER     NOT NULL,
    best_accuracy  REAL        NOT NULL,
    best_round     INTEGER     NOT NULL,
    final_accuracy REAL        NOT NULL,
    acc_at_r10     REAL,
    acc_at_r20     REAL,
    acc_at_r30     REAL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_SENSITIVITY_SQLITE = """
CREATE TABLE IF NOT EXISTS sensitivity_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment     TEXT NOT NULL,
    method         TEXT NOT NULL,
    seed           INTEGER NOT NULL,
    n_rounds       INTEGER NOT NULL,
    best_accuracy  REAL NOT NULL,
    best_round     INTEGER NOT NULL,
    final_accuracy REAL NOT NULL,
    acc_at_r10     REAL,
    acc_at_r20     REAL,
    acc_at_r30     REAL,
    created_at     TEXT NOT NULL
)
"""

METHODS = [
    ("FedAvg",  False),
    ("FedNova", True),
]


def _save_run_to_db(db_url, experiment, row):
    row = {**row, "experiment": experiment}
    if db_url:
        import sqlalchemy as sa
        engine = sa.create_engine(db_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(sa.text(_CREATE_SENSITIVITY_PG))
            conn.execute(sa.text("""
                INSERT INTO metrics.sensitivity_runs
                    (experiment, method, seed, n_rounds, best_accuracy, best_round,
                     final_accuracy, acc_at_r10, acc_at_r20, acc_at_r30)
                VALUES
                    (:experiment, :method, :seed, :n_rounds, :best_accuracy, :best_round,
                     :final_accuracy, :acc_at_r10, :acc_at_r20, :acc_at_r30)
            """), row)
    else:
        import sqlite3
        db_path = "checkpoints/sensitivity_results.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(_CREATE_SENSITIVITY_SQLITE)
            conn.execute(
                "INSERT INTO sensitivity_runs "
                "(experiment, method, seed, n_rounds, best_accuracy, best_round, "
                " final_accuracy, acc_at_r10, acc_at_r20, acc_at_r30, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (row["experiment"], row["method"], row["seed"], row["n_rounds"],
                 row["best_accuracy"], row["best_round"], row["final_accuracy"],
                 row.get("acc_at_r10"), row.get("acc_at_r20"), row.get("acc_at_r30"),
                 datetime.now(timezone.utc).isoformat()),
            )


def run_one(experiment, method_name, use_fednova, seed, n_rounds,
            client_loaders, test_loader, total, vocab, cal_loader):
    db_path = f"checkpoints/sensitivity_{method_name.lower()}_seed{seed}.db"
    store   = SQLiteCheckpointStore(db_path=db_path)

    logger.info(f"\n{'─'*60}")
    logger.info(f"[{method_name} | seed={seed} | rounds={n_rounds}]")
    logger.info(f"{'─'*60}")

    history, _ = run_federated_learning_manual(
        client_loaders            = client_loaders,
        test_loader               = test_loader,
        total_train_samples       = total,
        vocab                     = vocab,
        cal_loader                = cal_loader,
        override_num_rounds       = n_rounds,
        override_use_fednova      = use_fednova,
        override_random_seed      = seed,
        override_checkpoint_store = store,
        sensitivity_mode          = True,
    )

    accs      = history["accuracy"]
    best_acc  = max(accs)
    best_rnd  = history["rounds"][accs.index(best_acc)]
    final_acc = accs[-1]

    row = {
        "method":         method_name,
        "seed":           seed,
        "n_rounds":       n_rounds,
        "best_accuracy":  round(best_acc,  4),
        "best_round":     best_rnd,
        "final_accuracy": round(final_acc, 4),
        "acc_at_r10":     round(accs[9],   4) if len(accs) >= 10 else None,
        "acc_at_r20":     round(accs[19],  4) if len(accs) >= 20 else None,
        "acc_at_r30":     round(accs[29],  4) if len(accs) >= 30 else None,
    }

    _save_run_to_db(FL_DB_URL, experiment, row)
    logger.info(f"  → best={best_acc:.4f} (R{best_rnd}) | final={final_acc:.4f} | salvo no banco")
    return row


def summarize(method_name, runs):
    accs = np.array([r["best_accuracy"] for r in runs])
    return {
        "method":          method_name,
        "seeds":           [r["seed"] for r in runs],
        "best_accuracies": [r["best_accuracy"] for r in runs],
        "mean":  round(float(accs.mean()), 4),
        "std":   round(float(accs.std()),  4),
        "min":   round(float(accs.min()),  4),
        "max":   round(float(accs.max()),  4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30,
                        help="Rodadas por run (padrão: 30)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                        help="Seeds a testar (padrão: 42 7 123)")
    parser.add_argument("--experiment", type=str,
                        default=f"exp9_sensitivity_{datetime.now().strftime('%Y%m%d')}",
                        help="Nome do experimento para agrupar runs no banco")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ANÁLISE DE SENSIBILIDADE A SEED — FedAvg vs FedNova")
    logger.info(f"Experimento: {args.experiment}")
    logger.info(f"Seeds: {args.seeds} | Rounds por run: {args.rounds}")
    logger.info("=" * 60)

    if not FL_DB_URL:
        logger.error("FL_DB_URL não configurado.")
        sys.exit(1)

    logger.info("[1] Carregando dados do banco (uma única vez)...")
    client_loaders, test_loader, vocab, total, _, cal_loader = prepare_dataloaders_from_db(FL_DB_URL)
    logger.info(f"  Split fixo (RANDOM_SEED=42) | {total} treino | {len(test_loader.dataset)} teste")

    summaries = []
    for method_name, use_fednova in METHODS:
        runs = [
            run_one(args.experiment, method_name, use_fednova, seed, args.rounds,
                    client_loaders, test_loader, total, vocab, cal_loader)
            for seed in args.seeds
        ]
        summaries.append(summarize(method_name, runs))

    logger.info("\n" + "=" * 60)
    logger.info("RESULTADO CONSOLIDADO")
    logger.info("=" * 60)
    header = f"{'Método':<10} " + "  ".join(f"seed={s}" for s in args.seeds) + "   mean±std"
    logger.info(header)
    for s in summaries:
        accs  = "  ".join(f"{a:.4f}" for a in s["best_accuracies"])
        logger.info(f"{s['method']:<10} {accs}   {s['mean']:.4f} ± {s['std']:.4f}")

    delta_mean = summaries[1]["mean"] - summaries[0]["mean"]
    delta_std  = (summaries[0]["std"] ** 2 + summaries[1]["std"] ** 2) ** 0.5
    significant = abs(delta_mean) > delta_std
    logger.info(f"\n  Δ (FedNova − FedAvg): {delta_mean:+.4f} ± {delta_std:.4f}")
    logger.info(f"  {'Diferença significativa (> 1σ)' if significant else 'Diferença dentro do ruído (< 1σ)'}")

    # Resumo compacto — sem vetores, apenas estatísticas
    summary_out = {
        "experiment":   args.experiment,
        "config":       {"rounds": args.rounds, "seeds": args.seeds},
        "summary":      summaries,
        "delta_fednova_minus_fedavg": {
            "mean":        round(delta_mean, 4),
            "sigma":       round(delta_std,  4),
            "significant": significant,
        },
    }
    out_path = f"experiments/logs/seed_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).write_text(json.dumps(summary_out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nResumo: {out_path} | Log: {log_file}")
    logger.info("Dados detalhados: metrics.sensitivity_runs (banco)")


if __name__ == "__main__":
    main()
