#!/usr/bin/env python3
"""
run_experiments_simulation.py — Orquestrador da simulação MOSAIC-FL com dados sintéticos.

Demonstra o pipeline federado completo sem banco de dados real. Ideal para
apresentações, testes de ambiente e verificação do fluxo completo.
Para treinamento com dados reais FAPESP, use run_training.py.

Uso:
    python experiments/training_runner/run_experiments_simulation.py
    # ou via Makefile:
    make experiment
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/data", exist_ok=True)

log_file = os.environ.get(
    "FL_LOG_FILE",
    f"experiments/logs/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

from experiments.training.federated_training import FederatedTraining


def main() -> None:
    logger.info("=" * 60)
    logger.info("MOSAIC-FL — Simulação com Dados Sintéticos")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("=" * 60)
    logger.info(f"Log: {log_file}")

    ft = FederatedTraining(log_file=log_file, db_url=None, data_source="synthetic")

    logger.info("[1/5] Gerando dados sintéticos...")
    ft.load_synthetic()

    logger.info("[2/5] Aprendizado Federado (simulação)...")
    ft.train()

    logger.info("[3/5] Pipeline RAG...")
    try:
        rag_result = ft.run_rag()
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        rag_result = {"erro": str(e)}

    logger.info("[4/5] Baseline Random Forest...")
    try:
        baseline_result = ft.run_baseline()
    except Exception as e:
        logger.error(f"Erro no baseline RF: {e}")
        baseline_result = {"erro": str(e)}

    logger.info("[5/5] Ablation study — late fusion demográfica...")
    try:
        ablation_result = ft.run_ablation()
    except Exception as e:
        logger.error(f"Erro no ablation: {e}")
        ablation_result = {"erro": str(e)}

    ft.summarize(rag_result, baseline_result, ablation_result)


if __name__ == "__main__":
    main()
