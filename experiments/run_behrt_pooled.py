#!/usr/bin/env python3
"""
run_behrt_pooled.py — Pooled baseline para quantificação do custo de privacidade.

ARTEFATO DE PESQUISA — nunca executar em produção.

Treina SimplifiedBEHRT no pool completo BPSP+HSL para estabelecer um limite superior
de desempenho com a mesma arquitetura do modelo federado. Permite isolar o custo de
privacidade da federação de diferenças de arquitetura (BEHRT × BEHRT, não BEHRT × RF).

Uso:
    export FL_DB_URL='postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl'
    python experiments/run_behrt_pooled.py
    # ou via Makefile:
    make behrt-pooled

Saída:
    experiments/data/behrt_pooled_<timestamp>.json
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/data", exist_ok=True)

log_file = os.environ.get(
    "FL_LOG_FILE",
    f"experiments/logs/behrt_pooled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

from mosaicfl.core.config import FL_DB_URL, MODEL_CFG
from experiments.federated_training import (
    prepare_dataloaders_from_db,
    run_pooled_behrt,
    run_baseline_rf,
)


def main() -> None:
    logger.info("=" * 60)
    logger.info("MOSAIC-FL — BEHRT Pooled Baseline (Artefato de Pesquisa)")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("AVISO: Este script não deve ser executado em produção.")
    logger.info("=" * 60)

    if not FL_DB_URL:
        logger.error("FL_DB_URL não configurado.")
        logger.error("Configure: export FL_DB_URL='postgresql://user:pass@host:5432/db'")
        sys.exit(1)

    logger.info("[1/3] Carregando dados do banco (SequencePipeline)...")
    (
        client_loaders,
        test_loader,
        vocab,
        total,
        demographics_by_client,
        test_loader_demo,
        cal_loader,
    ) = prepare_dataloaders_from_db(FL_DB_URL)

    logger.info("[2/3] BEHRT pooled baseline (pool BPSP+HSL)...")
    pooled_result = run_pooled_behrt(
        client_loaders=client_loaders,
        test_loader=test_loader,
        demographics_by_client=demographics_by_client,
        test_loader_demo=test_loader_demo,
    )

    logger.info("[3/3] RF centralizado (referência comparativa de arquitetura)...")
    rf_result = run_baseline_rf(
        client_loaders=client_loaders,
        test_loader=test_loader,
        class_labels=list(MODEL_CFG.class_labels),
    )

    output = {
        "meta": {
            "script":    "run_behrt_pooled.py",
            "timestamp": datetime.now().isoformat(),
            "aviso":     "Artefato metodológico. Nunca implantar em produção.",
            "proposito": (
                "Quantifica o custo de privacidade da federação com mesma arquitetura. "
                "Comparar behrt_pooled_B_late_fusion com FL Config B do ablation study."
            ),
        },
        "behrt_pooled": pooled_result,
        "baseline_rf":  rf_result,
    }

    out_path = (
        Path("experiments/data")
        / f"behrt_pooled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Resultado salvo: {out_path}")

    _print_summary(pooled_result, rf_result)


def _print_summary(pooled: dict, rf: dict) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO — Custo de Privacidade da Federação (BEHRT × BEHRT)")
    logger.info("=" * 70)
    logger.info(f"{'Configuração':<45} {'Accuracy':>8} {'F1 Macro':>8}")
    logger.info("-" * 65)

    for key in ["behrt_pooled_A_sem_demo", "behrt_pooled_B_late_fusion"]:
        m = pooled.get(key, {})
        logger.info(
            f"{key:<45} "
            f"{m.get('accuracy', 'n/a'):>8} "
            f"{m.get('macro_f1', 'n/a'):>8}"
        )

    rf_c = rf.get("opcao_a_centralizado", {})
    logger.info(
        f"{'rf_centralizado (bag-of-tokens)':<45} "
        f"{rf_c.get('accuracy', 'n/a'):>8} "
        f"{rf_c.get('macro_f1', 'n/a'):>8}"
    )
    logger.info("-" * 65)
    logger.info(
        "Compare behrt_pooled_B_late_fusion com FL Config B (ablation study) "
        "para o custo real de privacidade (mesma arquitetura, mesmo split)."
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
