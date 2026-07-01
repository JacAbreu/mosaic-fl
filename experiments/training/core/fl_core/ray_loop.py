"""ray_loop.py — Aprendizado federado paralelo via Ray/Flower simulation."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import DEVICE, MODEL_CFG, NUM_ROUNDS
from mosaicfl.core.model import SimplifiedBEHRT

logger = logging.getLogger(__name__)


def run_federated_learning_ray(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL com Ray via flwr.simulation.start_simulation() — paralelo, rápido."""
    import flwr as fl

    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO COM RAY (PARALELA)")
    logger.info("=" * 60)
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {NUM_ROUNDS}")

    try:
        from flwr.simulation import start_simulation
    except ImportError as e:
        raise RuntimeError(
            "Ray não disponível. Instale com: pip install -U 'flwr[simulation]'"
        ) from e

    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        train_loader, val_loader = client_loaders[cid_int]
        return FedProxClient(cid_int, train_loader, val_loader)

    from experiments.training.experiment_server import start_server
    strategy, tracker, history = start_server(
        num_rounds=NUM_ROUNDS,
        num_clients=len(client_loaders),
        test_loader=test_loader,
        vocab=vocab or {},
    )

    logger.info(f"Rodando simulação Flower+Ray com {len(client_loaders)} clientes...")
    overall_start = time.time()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=len(client_loaders),
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0},
        )
    except StopIteration as e:
        logger.info(f"Convergência: {e}")
    except Exception as e:
        logger.error(f"Erro na simulação Ray: {e}")
        raise

    logger.info(f"Simulação concluída em {time.time() - overall_start:.2f}s")

    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    last_ckpt = history.get("last_checkpoint")
    if last_ckpt and Path(last_ckpt).exists():
        raw = torch.load(last_ckpt, map_location="cpu", weights_only=True)
        state_dict = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        global_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Modelo global restaurado de: {last_ckpt}")
    else:
        logger.warning("Checkpoint da simulação não encontrado — modelo com pesos aleatórios")

    hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Histórico salvo: {hist_path}")

    from mosaicfl.core.calibration import TemperatureScaler
    from mosaicfl.core.evaluation import evaluate, print_report

    try:
        report_raw = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=1.0)
        logger.info(f"Avaliação pré-calibração — ECE={report_raw.calibration.ece:.4f} "
                    f"AUC={report_raw.macro_auc:.4f} F1={report_raw.macro_f1:.4f}")
        print_report(report_raw)
    except Exception as exc:
        logger.warning(f"Avaliação pré-calibração falhou: {exc}")
        report_raw = None

    scaler = TemperatureScaler()
    try:
        scaler.fit(global_model, test_loader, device=str(DEVICE))
        logger.info(f"Calibração concluída — T={scaler.T:.4f}")
    except Exception as exc:
        logger.warning(f"Calibração falhou ({exc}) — T mantido em 1.0")

    try:
        report_cal = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=scaler.T)
        logger.info(f"Avaliação pós-calibração  — ECE={report_cal.calibration.ece:.4f} "
                    f"AUC={report_cal.macro_auc:.4f} F1={report_cal.macro_f1:.4f}")
        print_report(report_cal)
    except Exception as exc:
        logger.warning(f"Avaliação pós-calibração falhou: {exc}")
        report_cal = None

    eval_path = Path("experiments/logs") / f"evaluation_ray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import dataclasses
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps({
            "mode": "ray",
            "temperature": round(scaler.T, 4),
            "pre_calibration":  dataclasses.asdict(report_raw)  if report_raw  else None,
            "post_calibration": dataclasses.asdict(report_cal) if report_cal else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Relatório de avaliação salvo: {eval_path}")
    except Exception as exc:
        logger.warning(f"Falha ao salvar relatório de avaliação: {exc}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(
        {"model_state": global_model.state_dict(), "vocab": vocab or {}, "temperature": scaler.T},
        ckpt_path,
    )
    logger.info(f"Checkpoint salvo: {ckpt_path} (vocab_size={len(vocab or {})}, T={scaler.T:.4f})")

    return history, global_model
