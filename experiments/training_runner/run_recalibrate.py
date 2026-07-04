#!/usr/bin/env python3
"""
run_recalibrate.py — Re-calibração de temperatura sobre o melhor checkpoint salvo.

Útil quando a calibração falhou em um experimento anterior (ex: T negativo por bug
no TemperatureScaler) sem necessidade de re-executar o treinamento completo.

Carrega o melhor checkpoint do PostgreSQL, roda TemperatureScaler.fit() com o
código corrigido e salva o relatório de avaliação corrigido.

Uso:
    export FL_DB_URL='postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl'
    python experiments/training_runner/run_recalibrate.py
    # ou via Makefile:
    make recalibrate
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/data", exist_ok=True)

log_file = os.environ.get(
    "FL_LOG_FILE",
    f"experiments/logs/recalibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

from mosaicfl.core.calibration import TemperatureScaler
from mosaicfl.core.config import DEVICE, FL_DB_URL, MODEL_CFG
from mosaicfl.core.evaluation import evaluate, print_report
from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.shared.checkpoint_store import get_checkpoint_store
from experiments.training.core.dataloaders import prepare_dataloaders_from_db


def main() -> None:
    logger.info("=" * 60)
    logger.info("MOSAIC-FL — Re-calibração do melhor checkpoint")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("=" * 60)

    if not FL_DB_URL:
        logger.error("FL_DB_URL não configurado.")
        sys.exit(1)

    logger.info("[1/4] Carregando dados do banco...")
    (
        _client_loaders,
        test_loader,
        _vocab,
        _total,
        _demographics_by_client,
        _test_loader_demo,
        cal_loader,
        _test_loader_origin,
        _origin_labels,
    ) = prepare_dataloaders_from_db(FL_DB_URL)
    logger.info("Dados carregados — test=%d | cal=%d amostras",
                len(test_loader.dataset), len(cal_loader.dataset))

    logger.info("[2/4] Carregando melhor checkpoint do banco (load_best)...")
    store = get_checkpoint_store(FL_DB_URL)
    ckpt = store.load_best()
    if ckpt is None:
        logger.error("Nenhum checkpoint encontrado no banco.")
        sys.exit(1)

    model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Checkpoint carregado — round=%s | model_version=%s",
                ckpt.get("checkpoint_round", "?"),
                ckpt.get("model_version", "?"))

    logger.info("[3/4] Avaliação pré-calibração...")
    report_pre = evaluate(
        model, test_loader,
        class_labels=MODEL_CFG.class_labels,
        device=str(DEVICE),
        temperature=1.0,
    )
    logger.info("Pré-calibração — Acc=%.4f ECE=%.4f AUC=%.4f F1=%.4f",
                report_pre.accuracy, report_pre.calibration.ece,
                report_pre.macro_auc, report_pre.macro_f1)
    print_report(report_pre)

    logger.info("[4/4] Calibração de temperatura (código corrigido — log-space)...")
    scaler = TemperatureScaler()
    scaler.fit(model, cal_loader, device=str(DEVICE))
    logger.info("T=%.4f (positivo garantido por parametrização em log-space)", scaler.T)

    report_cal = evaluate(
        model, test_loader,
        class_labels=MODEL_CFG.class_labels,
        device=str(DEVICE),
        temperature=scaler.T,
    )
    logger.info("Pós-calibração  — Acc=%.4f ECE=%.4f AUC=%.4f F1=%.4f",
                report_cal.accuracy, report_cal.calibration.ece,
                report_cal.macro_auc, report_cal.macro_f1)
    print_report(report_cal)

    import dataclasses
    out_path = Path("experiments/logs") / f"recalibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "checkpoint_round": ckpt.get("checkpoint_round"),
        "model_version":    ckpt.get("model_version"),
        "temperature":      round(scaler.T, 4),
        "pre_calibration":  dataclasses.asdict(report_pre),
        "post_calibration": dataclasses.asdict(report_cal),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Relatório salvo: %s", out_path)
    logger.info("RECALIBRACAO_COMPLETA T=%.4f ECE_pre=%.4f ECE_pos=%.4f",
                scaler.T, report_pre.calibration.ece, report_cal.calibration.ece)


if __name__ == "__main__":
    main()
