#!/usr/bin/env python3
"""
run_training.py — Orquestrador do treinamento federado com dados reais FAPESP.

Requer FL_DB_URL configurado. Não há fallback sintético — se o banco não estiver
acessível, o script aborta. Para demonstração com dados sintéticos, use
run_experiments_simulation.py.

Uso:
    export FL_DB_URL='postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl'
    python experiments/training_runner/run_training.py
    # ou via Makefile:
    make training
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
    f"experiments/logs/run_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

from mosaicfl.core.config import DEVICE, FL_DB_URL, FL_ENV
from experiments.training.federated_training import FederatedTraining


def main() -> None:
    logger.info("=" * 60)
    logger.info("MOSAIC-FL — Treinamento Federado com Dados Reais FAPESP")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("=" * 60)
    logger.info(f"Ambiente: {FL_ENV} | Log: {log_file}")
    logger.info(f"Device: {DEVICE}")

    if not FL_DB_URL:
        logger.error("FL_DB_URL não configurado — treinamento real requer banco de dados.")
        logger.error("Configure: export FL_DB_URL='postgresql://user:pass@host:5432/db'")
        logger.error("Para simulação sem banco, use: python experiments/training_runner/run_experiments_simulation.py")
        sys.exit(1)

    ft = FederatedTraining(log_file=log_file, db_url=FL_DB_URL, data_source="fapesp")

    logger.info("[1/5] Carregando dados do banco (SequencePipeline)...")
    ft.load_from_db(FL_DB_URL)

    logger.info("[2/5] Aprendizado Federado...")
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
