#!/usr/bin/env python3
"""
federated_training.py — Lógica central do MOSAIC-FL.

Contém todas as funções de treinamento federado, RAG, baseline e ablation.
Importado pelos orquestradores:
  - run_training.py          → dados reais FAPESP (banco)
  - run_experiments_simulation.py → dados sintéticos (demonstração)
"""
import os
import sys
import json
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

import flwr as fl

from mosaicfl.core.config import *
from mosaicfl.core.data_loader import load_clinical_dataset, diagnose_dataset, load_with_fallback
from infrastructure.shared.checkpoint_store import get_checkpoint_store
from infrastructure.shared.metrics_store import get_metrics_store
from mosaicfl.core.preprocessor import EHRPreprocessor, split_by_institution, SequencePipeline
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.client import FedProxClient
from experiments.experiment_server import start_server
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.rag import ClinicalRAG
from mosaicfl.core.interpretability import BEHRTPatternExtractor

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DOS DATALOADERS — MODO SINTÉTICO / CSV
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataloaders(
    df_raw: pd.DataFrame,
    preprocessor: EHRPreprocessor,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[Dict, DataLoader, Dict, int]:
    """Pré-processa e cria DataLoaders para FL."""

    text_cols = ["sintoma", "exame", "diagnostico"]
    df_proc, summary = preprocessor.process(df_raw, text_cols=text_cols)
    logger.info(f"Pré-processamento: {summary['tamanho_vocabulario']} tokens | "
                f"{summary['total_amostras']} amostras")

    client_dfs = split_by_institution(
        df_proc,
        institution_col="instituicao",
        num_clients=NUM_CLIENTS,
        stratify_col="desfecho",
        random_state=RANDOM_SEED,
    )

    seq_cols = [c for c in df_proc.columns if c.endswith("_encoded")]
    if not seq_cols:
        raise ValueError("Nenhuma coluna '_encoded' encontrada após pré-processamento.")

    def make_sequences(sub_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = [], []
        for _, row in sub_df.iterrows():
            seq = [int(row[c]) for c in seq_cols if pd.notna(row[c])]
            seq = seq[:max_seq_len] if len(seq) >= max_seq_len else seq + [0] * (max_seq_len - len(seq))
            sequences.append(seq)
            labels.append(int(row["desfecho"]))
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    client_loaders = {}
    total_train_samples = 0

    for cid, subset in client_dfs.items():
        if len(subset) < 10:
            logger.warning(f"Cliente {cid}: apenas {len(subset)} amostras — pulando.")
            continue

        n_train = int(0.8 * len(subset))
        train_x, train_y = make_sequences(subset.iloc[:n_train])
        val_x, val_y = make_sequences(subset.iloc[n_train:])

        client_loaders[cid] = (
            DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size),
        )
        total_train_samples += len(train_x)
        logger.info(f"Cliente {cid}: {len(train_x)} treino | {len(val_x)} val")

    if not client_loaders:
        raise ValueError("Nenhum cliente válido criado.")

    test_size = max(10, int(0.2 * len(df_proc)))
    test_x, test_y = make_sequences(df_proc.sample(n=test_size, random_state=RANDOM_SEED))
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    logger.info(f"Teste global: {len(test_x)} amostras")

    return client_loaders, test_loader, preprocessor.vocab_map, total_train_samples


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DOS DATALOADERS — MODO BANCO DE DADOS (TENSORES REAIS)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataloaders_from_db(
    db_url: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[Dict, DataLoader, Dict, int]:
    """
    Constrói DataLoaders para FL a partir dos tensores reais do SequencePipeline.

    Divide por hospital: cada hospital vira um cliente FL.
    Split: 70% treino / 10% val / 10% cal (calibração) / 10% teste global.
    cal é reservado exclusivamente para temperature scaling — nunca exposto ao FL.

    Returns:
        client_loaders, test_loader, vocab, total_train_samples,
        demographics_by_client, test_loader_demo, cal_loader
    """
    logger.info("[db] Iniciando carregamento via SequencePipeline...")
    pipeline = SequencePipeline(connection_string=db_url, max_seq_len=MAX_SEQ_LEN)
    hospital_data = pipeline.build_per_hospital()

    if not hospital_data:
        raise RuntimeError("build_per_hospital() retornou vazio — sem dados na base.")

    vocab = next(iter(hospital_data.values()))[2]

    rng = torch.Generator()
    rng.manual_seed(RANDOM_SEED)

    client_loaders: Dict = {}
    demographics_by_client: Dict = {}
    test_seqs_list, test_lbls_list, test_demo_list = [], [], []
    cal_seqs_list, cal_lbls_list = [], []
    total_train_samples = 0

    for cid, (hospital_id, (seqs, labels, _, demo)) in enumerate(hospital_data.items()):
        n = len(seqs)
        if n < 10:
            logger.warning(f"[db] Hospital {hospital_id}: apenas {n} amostras — pulando.")
            continue

        perm = torch.randperm(n, generator=rng)
        n_train = int(0.7 * n)
        n_val   = int(0.1 * n)
        n_cal   = int(0.1 * n)

        train_seqs = seqs[perm[:n_train]]
        train_lbls = labels[perm[:n_train]]
        train_demo = demo[perm[:n_train]]
        val_seqs   = seqs[perm[n_train:n_train + n_val]]
        val_lbls   = labels[perm[n_train:n_train + n_val]]
        val_demo   = demo[perm[n_train:n_train + n_val]]
        cal_seqs   = seqs[perm[n_train + n_val:n_train + n_val + n_cal]]
        cal_lbls   = labels[perm[n_train + n_val:n_train + n_val + n_cal]]
        test_seqs_list.append(seqs[perm[n_train + n_val + n_cal:]])
        test_lbls_list.append(labels[perm[n_train + n_val + n_cal:]])
        test_demo_list.append(demo[perm[n_train + n_val + n_cal:]])

        client_loaders[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_seqs, val_lbls), batch_size=batch_size),
        )
        demographics_by_client[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls, train_demo), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_seqs, val_lbls, val_demo), batch_size=batch_size),
        )
        cal_seqs_list.append(cal_seqs)
        cal_lbls_list.append(cal_lbls)
        total_train_samples += len(train_seqs)
        logger.info(
            f"[db] Hospital {hospital_id} → cliente {cid}: "
            f"{len(train_seqs)} treino | {len(val_seqs)} val | {len(cal_seqs)} cal | "
            f"age_mean={train_demo[:, 0].mean():.2f} sex_M={int((train_demo[:, 1] == 1.0).sum())}"
        )

    if not client_loaders:
        raise RuntimeError("Nenhum cliente válido criado a partir dos dados reais.")

    test_seqs  = torch.cat(test_seqs_list, dim=0)
    test_lbls  = torch.cat(test_lbls_list, dim=0)
    test_demo  = torch.cat(test_demo_list, dim=0)
    cal_seqs_all = torch.cat(cal_seqs_list, dim=0)
    cal_lbls_all = torch.cat(cal_lbls_list, dim=0)

    test_loader      = DataLoader(TensorDataset(test_seqs, test_lbls), batch_size=batch_size)
    test_loader_demo = DataLoader(TensorDataset(test_seqs, test_lbls, test_demo), batch_size=batch_size)
    cal_loader       = DataLoader(TensorDataset(cal_seqs_all, cal_lbls_all), batch_size=batch_size)

    logger.info(
        f"[db] Teste global: {len(test_seqs)} amostras | Cal global: {len(cal_seqs_all)} amostras "
        f"| {len(client_loaders)} clientes FL"
    )

    return client_loaders, test_loader, vocab, total_train_samples, demographics_by_client, test_loader_demo, cal_loader


# ═══════════════════════════════════════════════════════════════════════════════
# AGREGAÇÃO E AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_fedavg(state_dicts: List[OrderedDict], weights: List[int]) -> OrderedDict:
    """Agrega state_dicts via média ponderada (FedAvg)."""
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Peso total de agregação é zero.")

    global_state = OrderedDict()
    for key in state_dicts[0].keys():
        global_state[key] = torch.zeros_like(state_dicts[0][key].float())

    for state_dict, weight in zip(state_dicts, weights):
        for key in state_dict.keys():
            global_state[key] += state_dict[key].float() * (weight / total_weight)

    return global_state


def evaluate_global_model(model: SimplifiedBEHRT, test_loader: DataLoader) -> Tuple[float, float]:
    """Avalia modelo global no conjunto de teste."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss_sum += loss.item() * batch_y.size(0)
            _, predicted = torch.max(logits, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = loss_sum / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════════
# MODO 1: SIMULAÇÃO MANUAL (SEM RAY)
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning_manual(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
    cal_loader: DataLoader = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL manualmente, sem Ray — sequencial, leve, didático."""
    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO MANUAL (SEM RAY)")
    logger.info("=" * 60)
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {NUM_ROUNDS}")
    logger.info(f"Mu: {PROXIMAL_MU} | Batch: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS}")

    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    global_state = global_model.state_dict()

    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "communication_mb": [],
        "client_losses": [],
    }

    stable_count = 0
    prev_accuracy = 0.0
    converged_round = None

    overall_start = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"RODADA {round_num}/{NUM_ROUNDS}")
        logger.info(f"{'='*60}")

        client_states = []
        client_weights = []
        client_metrics = []

        for cid, (train_loader, val_loader) in client_loaders.items():
            logger.info(f"  [Cliente {cid}] Treinando...")

            client = FedProxClient(cid, train_loader, val_loader)
            global_params = [v.cpu().numpy() for v in global_state.values()]
            client.set_parameters(global_params)

            fit_params, num_samples, metrics = client.fit(
                global_params, config={"current_round": round_num}
            )

            fit_state = OrderedDict()
            for k, v in zip(global_state.keys(), fit_params):
                fit_state[k] = torch.tensor(v)

            client_states.append(fit_state)
            client_weights.append(num_samples)
            client_metrics.append(metrics)

            logger.info(f"  [Cliente {cid}] Loss: {metrics['loss']:.4f} | Amostras: {num_samples}")

        logger.info(f"  [Servidor] Agregando pesos de {len(client_states)} clientes...")
        global_state = aggregate_fedavg(client_states, client_weights)
        global_model.load_state_dict(global_state)

        loss_global, acc_global = evaluate_global_model(global_model, test_loader)
        round_duration = time.time() - round_start

        model_size_mb = sum(p.numel() * 4 for p in global_model.parameters()) / (1024 ** 2)
        comm_mb = model_size_mb * len(client_states) * 2

        logger.info(f"  [Servidor] Rodada {round_num} em {round_duration:.2f}s | Loss: {loss_global:.4f} | Acc: {acc_global:.4f}")

        history["rounds"].append(round_num)
        history["accuracy"].append(acc_global)
        history["loss"].append(loss_global)
        history["communication_mb"].append(comm_mb)
        history["client_losses"].append([m["loss"] for m in client_metrics])

        if round_num > 1:
            delta = abs(acc_global - prev_accuracy)
            if delta < CONVERGENCE_THRESHOLD:
                stable_count += 1
                logger.info(f"  [Convergência] Δ={delta:.5f} < {CONVERGENCE_THRESHOLD} ({stable_count}/{CONVERGENCE_PATIENCE})")
            else:
                stable_count = 0
                logger.info(f"  [Convergência] Δ={delta:.5f} > {CONVERGENCE_THRESHOLD} (reset)")

            if stable_count >= CONVERGENCE_PATIENCE and converged_round is None:
                converged_round = round_num
                logger.info(f"\nCONVERGENCIA ATINGIDA na rodada {round_num}!")
                break

        prev_accuracy = acc_global

    overall_duration = time.time() - overall_start

    logger.info(f"\n{'='*60}")
    logger.info("SIMULAÇÃO CONCLUÍDA")
    logger.info(f"{'='*60}")
    logger.info(f"  Rodadas:      {round_num}")
    logger.info(f"  Convergência: {'Sim (rodada ' + str(converged_round) + ')' if converged_round else 'Não'}")
    logger.info(f"  Acurácia:     {history['accuracy'][-1]:.4f}")
    logger.info(f"  Loss:         {history['loss'][-1]:.4f}")
    logger.info(f"  Tráfego:      {sum(history['communication_mb']):.2f} MB")
    logger.info(f"  Tempo total:  {overall_duration:.2f}s ({overall_duration/60:.2f} min)")
    logger.info(f"{'='*60}")

    logger.info(
        "FL_TRAINING_COMPLETE rounds=%d converged=%s accuracy=%.4f loss=%.4f "
        "duration_s=%.1f traffic_mb=%.2f",
        round_num,
        bool(converged_round),
        history["accuracy"][-1],
        history["loss"][-1],
        overall_duration,
        sum(history["communication_mb"]),
    )

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

    _calib_loader = cal_loader if cal_loader is not None else test_loader
    if cal_loader is None:
        logger.warning("calibration_set_fallback: cal_loader ausente — usando test_loader (modo sintético)")
    scaler = TemperatureScaler()
    try:
        scaler.fit(global_model, _calib_loader, device=str(DEVICE))
        n_cal = len(_calib_loader.dataset) if hasattr(_calib_loader, "dataset") else "?"
        logger.info(f"Calibração concluída — T={scaler.T:.4f} (cal_set={n_cal} amostras)")
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

    eval_path = Path("experiments/logs") / f"evaluation_round_{round_num}.json"
    try:
        import dataclasses
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps({
            "round": round_num,
            "temperature": round(scaler.T, 4),
            "pre_calibration":  dataclasses.asdict(report_raw)  if report_raw  else None,
            "post_calibration": dataclasses.asdict(report_cal) if report_cal else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Relatório de avaliação salvo: {eval_path}")
    except Exception as exc:
        logger.warning(f"Falha ao salvar relatório de avaliação: {exc}")

    checkpoint_store = get_checkpoint_store(FL_DB_URL)
    checkpoint_store.save(
        round_num=round_num,
        state_dict=global_model.state_dict(),
        vocab=vocab or {},
        accuracy=history["accuracy"][-1] if history["accuracy"] else 0.0,
        loss=history["loss"][-1] if history["loss"] else 0.0,
        temperature=scaler.T,
    )
    logger.info(f"Checkpoint salvo no store ({type(checkpoint_store).__name__}, vocab_size={len(vocab or {})}, T={scaler.T:.4f})")

    return history, global_model


# ═══════════════════════════════════════════════════════════════════════════════
# MODO 2: SIMULAÇÃO COM RAY (PARALELA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning_ray(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL com Ray via flwr.simulation.start_simulation() — paralelo, rápido."""
    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO COM RAY (PARALELA)")
    logger.info("=" * 60)
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {NUM_ROUNDS}")

    try:
        from flwr.simulation import start_simulation
    except ImportError as e:
        logger.error("RAY NÃO ESTÁ INSTALADO!")
        logger.error('Para usar o modo paralelo: pip install -U "flwr[simulation]"')
        logger.error("Ou edite mosaicfl/core/config.py: USE_RAY = False")
        raise RuntimeError("Ray não disponível. Instale com: pip install -U 'flwr[simulation]'") from e

    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        train_loader, val_loader = client_loaders[cid_int]
        return FedProxClient(cid_int, train_loader, val_loader)

    from experiments.experiment_server import start_server
    from mosaicfl.core.federated import weighted_average
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

    overall_end = time.time()
    logger.info(f"Simulação concluída em {overall_end - overall_start:.2f}s")

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
    torch.save({"model_state": global_model.state_dict(), "vocab": vocab or {}, "temperature": scaler.T}, ckpt_path)
    logger.info(f"Checkpoint salvo: {ckpt_path} (vocab_size={len(vocab or {})}, T={scaler.T:.4f})")

    return history, global_model


# ═══════════════════════════════════════════════════════════════════════════════
# ROTEADOR FL — seleciona manual ou Ray conforme USE_RAY
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
    cal_loader: DataLoader = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """
    Roteia para o modo correto baseado em config.USE_RAY.

    Para alternar:
        Edite mosaicfl/core/config.py → USE_RAY = False  (manual, leve)
        Edite mosaicfl/core/config.py → USE_RAY = True   (Ray, paralelo)
    """
    use_ray = bool(USE_RAY)

    if use_ray:
        logger.info("Modo Ray ativado (USE_RAY=True).")
        return run_federated_learning_ray(client_loaders, test_loader, total_train_samples, vocab=vocab)
    else:
        logger.info("Modo manual ativado (USE_RAY=False). Ray NÃO é necessário.")
        return run_federated_learning_manual(client_loaders, test_loader, total_train_samples, vocab=vocab, cal_loader=cal_loader)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RAG
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_rag_precision_at_k(
    rag: "ClinicalRAG",
    test_loader: DataLoader,
    vocab_inverse: Dict[int, str],
    class_labels: List[str],
    k: int = 3,
) -> Dict:
    """
    Avalia a qualidade da recuperação do RAG via Precision@k.

    Para cada amostra do test_loader, consulta o RAG com os tokens do paciente
    e verifica quantos dos k casos recuperados têm o mesmo desfecho que o rótulo
    real. Métrica central para CDSS humano-no-loop.
    """
    hits_total = 0
    queries_total = 0
    per_class_hits: Dict[str, int] = {lbl: 0 for lbl in class_labels}
    per_class_queries: Dict[str, int] = {lbl: 0 for lbl in class_labels}

    for batch_x, batch_y in test_loader:
        for seq, label_idx in zip(batch_x.tolist(), batch_y.tolist()):
            tokens = [
                vocab_inverse[t]
                for t in seq
                if t > 2 and t in vocab_inverse
            ]
            if not tokens:
                continue

            ground_truth = (
                class_labels[label_idx]
                if label_idx < len(class_labels)
                else f"classe_{label_idx}"
            )
            query = ", ".join(tokens[:20])
            retrieved = rag.retrieve(query, top_k=k)

            n_hits = sum(
                1 for c in retrieved
                if c.get("metadata", {}).get("desfecho") == ground_truth
            )
            hits_total += n_hits
            queries_total += k
            per_class_hits[ground_truth] = per_class_hits.get(ground_truth, 0) + n_hits
            per_class_queries[ground_truth] = per_class_queries.get(ground_truth, 0) + k

    precision_at_k = round(hits_total / queries_total, 4) if queries_total > 0 else 0.0
    per_class_precision = {
        lbl: round(per_class_hits[lbl] / per_class_queries[lbl], 4)
        if per_class_queries[lbl] > 0 else None
        for lbl in class_labels
    }

    logger.info(f"RAG Precision@{k} (recuperação): {precision_at_k:.4f}")
    for lbl, p in per_class_precision.items():
        p_str = f"{p:.4f}" if p is not None else "n/a"
        logger.info(f"  {lbl}: {p_str}")

    return {
        f"precision_at_{k}": precision_at_k,
        f"per_class_precision_at_{k}": per_class_precision,
        "k": k,
        "n_queries": queries_total // k,
    }


def run_rag_pipeline(
    global_model: SimplifiedBEHRT,
    vocab_map: Dict,
    test_loader: DataLoader,
) -> Dict:
    """Extrai padrões do BEHRT, gera justificativa via RAG e avalia Precision@k."""

    logger.info("=" * 60)
    logger.info("PIPELINE RAG")
    logger.info("=" * 60)

    all_labels = []
    for _, batch_y in test_loader:
        all_labels.extend(batch_y.tolist())
    desfechos = sorted(set(all_labels))
    logger.info(f"Desfechos presentes no test_loader: {desfechos}")

    extractor = BEHRTPatternExtractor(global_model, vocab_map)
    patterns = extractor.generate_all_profiles(test_loader, desfechos=desfechos)
    logger.info(f"Padrões extraídos: {len(patterns)} perfis")

    rag = ClinicalRAG()
    rag.build_knowledge_base(patterns)

    vocab_inverse = {v: k for k, v in vocab_map.items()}
    labels = MODEL_CFG.class_labels

    logger.info("Avaliando Precision@k da recuperação...")
    precision_metrics = _eval_rag_precision_at_k(
        rag, test_loader, vocab_inverse, list(labels), k=3
    )

    sample_label = desfechos[0]
    sample_tokens: List[str] = []
    for batch_x, batch_y in test_loader:
        raw_tokens = [
            vocab_inverse.get(t, "")
            for t in batch_x[0].tolist()
            if t > 2
        ]
        sample_tokens = [t for t in raw_tokens if t][:10]
        sample_label = int(batch_y[0].item())
        break

    label_name = (
        labels[sample_label] if sample_label < len(labels) else f"classe_{sample_label}"
    )
    patient_data = {
        "tokens": ", ".join(sample_tokens) if sample_tokens else "dados laboratoriais",
    }
    model_prediction = {
        "diagnostico": label_name,
        "probabilidade": random.uniform(0.55, 0.95),
    }

    result = rag.explain(patient_data, model_prediction)
    logger.info(f"Justificativa — confiável: {result['confiavel']} | "
                f"alucinação: {result['alucinacao_detectada']}")

    result["precision_metrics"] = precision_metrics

    rag_path = f"experiments/data/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARATIVO — Random Forest (Bag-of-Tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def _loader_to_bow(
    loader: DataLoader,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converte sequências de token IDs de um DataLoader em vetores Bag-of-Tokens."""
    X, y = [], []
    for batch_x, batch_y in loader:
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
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
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
        n_estimators=200,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
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
        f"AUC={auc_str}  "
        f"F1={m_a['macro_f1']:.4f}  "
        f"ECE={m_a['ece']:.4f}"
    )

    logger.info("\n[Opção B] RF por Hospital — modelo independente por cliente")
    per_client: Dict = {}
    for cid, (train_loader, _) in client_loaders.items():
        X_c, y_c = _loader_to_bow(train_loader, vocab_size)
        n_local_classes = int(len(np.unique(y_c)))
        rf_local = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
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
            f"  Hospital {cid}: Acc={m['accuracy']:.4f}  "
            f"AUC={auc_h}  "
            f"F1={m['macro_f1']:.4f}  "
            f"(treino={len(X_c)}, classes_locais={n_local_classes})"
        )
    results["opcao_b_por_hospital"] = {
        "per_client": per_client,
        "descricao": "RF independente por hospital — sem compartilhamento de dados",
    }

    header = f"\n{'Modelo':<42} {'Accuracy':>8} {'AUC-ROC':>8} {'F1 Macro':>8} {'ECE':>7}"
    sep    = "-" * 76
    logger.info("\n" + "=" * 60)
    logger.info("TABELA COMPARATIVA — Baseline vs. BEHRT (FL)")
    logger.info("=" * 60)
    logger.info(header)
    logger.info(sep)
    def _fmt(v: Optional[float]) -> str:
        return f"{v:>8.4f}" if v is not None else "     n/a"

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
    logger.info(sep)
    logger.info(f"{'SimplifiedBEHRT (FL) — ver evaluation_round_*.json':<42}")
    logger.info("=" * 60)

    results["meta"] = {
        "vocab_size":   vocab_size,
        "n_test":       int(len(X_test)),
        "class_labels": class_labels,
        "rf_params":    {"n_estimators": 200, "class_weight": "balanced"},
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION — LATE FUSION DEMOGRÁFICA (age_norm + sex_binary)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_correlated_demo(labels: torch.Tensor, noise_std: float = 0.15, seed: int = 42) -> torch.Tensor:
    """
    Gera demográficos sintéticos correlacionados com os labels de prognóstico.
    Baseia-se no perfil epidemiológico do COVID-19.
    """
    rng = np.random.default_rng(seed)
    age_means = {0: 0.45, 1: 0.55, 2: 0.65, 3: 0.70}
    sex_probs  = {0: 0.45, 1: 0.50, 2: 0.60, 3: 0.65}

    ages  = np.zeros(len(labels), dtype=np.float32)
    sexes = np.zeros(len(labels), dtype=np.float32)
    for i, lbl in enumerate(labels.tolist()):
        lbl = min(int(lbl), 3)
        ages[i]  = float(np.clip(rng.normal(age_means[lbl], noise_std), 0.18, 0.95))
        sexes[i] = 1.0 if rng.random() < sex_probs[lbl] else 0.0

    return torch.from_numpy(np.stack([ages, sexes], axis=1))


def run_ablation_demographics(
    client_loaders: Dict,
    test_loader: DataLoader,
    demographics_by_client: Optional[Dict] = None,
    test_loader_demo: Optional[DataLoader] = None,
    n_epochs: int = 10,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Ablation study: late fusion demográfica (age_norm + sex_binary) no SimplifiedBEHRT.

    Config A — sequências de exames apenas (demo_dim=0, modelo base)
    Config B — sequências + late fusion demográfica (demo_dim=2)

    Com dados reais: usa age_norm e sex_binary do SequencePipeline.
    Com dados sintéticos: gera demográficos correlacionados com os labels.
    """
    from mosaicfl.core.model import SimplifiedBEHRT as _BEHRT
    from mosaicfl.core.evaluation import evaluate, print_report

    random_seed = random_seed if random_seed is not None else RANDOM_SEED
    use_real_demo = demographics_by_client is not None

    logger.info("=" * 60)
    logger.info("ABLATION — Late Fusion Demográfica")
    logger.info(f"  Fonte dos demográficos: {'dados reais (FAPESP)' if use_real_demo else 'sintéticos correlacionados'}")
    logger.info("  Config A: SimplifiedBEHRT sem demográficos (demo_dim=0)")
    logger.info("  Config B: SimplifiedBEHRT + late fusion (demo_dim=2)")
    logger.info(f"  Épocas locais: {n_epochs} | seed: {random_seed}")
    logger.info("=" * 60)

    def _collect_labels(loaders: Dict) -> torch.Tensor:
        lbls = []
        for _, (train_loader, _) in loaders.items():
            for _, batch_y in train_loader:
                lbls.append(batch_y)
        return torch.cat(lbls, dim=0)

    def _train_local(model: nn.Module, loaders: Dict, with_demo: bool, demo_loaders: Optional[Dict]) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        model.train()

        synth_demo_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if with_demo and not use_real_demo:
            for cid, (train_loader, _) in loaders.items():
                all_x, all_y = [], []
                for bx, by in train_loader:
                    all_x.append(bx); all_y.append(by)
                all_x = torch.cat(all_x); all_y = torch.cat(all_y)
                synth_demo_cache[cid] = (all_x, all_y, _make_correlated_demo(all_y, seed=random_seed + cid))

        for _ in range(n_epochs):
            if with_demo and use_real_demo and demo_loaders:
                for cid, (demo_train_loader, _) in demo_loaders.items():
                    for batch_x, batch_y, batch_demo in demo_train_loader:
                        batch_x = batch_x.to(DEVICE)
                        batch_y = batch_y.to(DEVICE)
                        batch_demo = batch_demo.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(batch_x, demographics=batch_demo)
                        criterion(logits, batch_y).backward()
                        optimizer.step()
            elif with_demo and not use_real_demo:
                for cid, (all_x, all_y, all_demo) in synth_demo_cache.items():
                    dataset = TensorDataset(all_x, all_y, all_demo)
                    for bx, by, bd in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
                        bx = bx.to(DEVICE); by = by.to(DEVICE); bd = bd.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(bx, demographics=bd)
                        criterion(logits, by).backward()
                        optimizer.step()
            else:
                for cid, (train_loader, _) in loaders.items():
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(DEVICE); batch_y = batch_y.to(DEVICE)
                        optimizer.zero_grad()
                        criterion(model(batch_x), batch_y).backward()
                        optimizer.step()

    def _eval_with_demo(model: nn.Module, t_loader_demo: DataLoader) -> Dict:
        model.eval()
        all_preds, all_labels = [], []
        criterion = nn.CrossEntropyLoss()
        total_loss, total_n = 0.0, 0

        with torch.no_grad():
            for batch in t_loader_demo:
                if len(batch) == 3:
                    bx, by, bd = batch
                else:
                    bx, by = batch
                    bd = _make_correlated_demo(by, seed=random_seed + 99)
                bx = bx.to(DEVICE); by = by.to(DEVICE); bd = bd.to(DEVICE)
                logits = model(bx, demographics=bd)
                total_loss += criterion(logits, by).item() * by.size(0)
                total_n += by.size(0)
                all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
                all_labels.extend(by.cpu().tolist())

        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
            "macro_f1": round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
            "loss":     round(total_loss / total_n, 4) if total_n > 0 else None,
        }

    results: Dict = {}

    for config_name, demo_dim, with_demo in [
        ("config_A_sem_demo",    0, False),
        ("config_B_late_fusion", 2, True),
    ]:
        logger.info(f"\n[Ablação] Treinando {config_name} (demo_dim={demo_dim}, épocas={n_epochs})...")
        torch.manual_seed(random_seed)
        model = _BEHRT(use_cls_token=True, demo_dim=demo_dim).to(DEVICE)

        _train_local(model, client_loaders, with_demo, demographics_by_client)

        if with_demo:
            t_loader = test_loader_demo if (use_real_demo and test_loader_demo is not None) else test_loader
            metrics = _eval_with_demo(model, t_loader)
            metrics["descricao"] = (
                "SimplifiedBEHRT + late fusion demográfica "
                f"({'dados reais FAPESP' if use_real_demo else 'sintéticos correlacionados'})"
            )
        else:
            try:
                report = evaluate(model, test_loader, class_labels=MODEL_CFG.class_labels,
                                  device=str(DEVICE), temperature=1.0)
                metrics = {
                    "accuracy":  round(report.accuracy, 4),
                    "macro_f1":  round(report.macro_f1, 4),
                    "macro_auc": round(report.macro_auc, 4) if report.macro_auc else None,
                    "ece":       round(report.calibration.ece, 4),
                    "descricao": "SimplifiedBEHRT sem demográficos (baseline)",
                }
            except Exception as exc:
                logger.warning(f"evaluate() falhou para {config_name}: {exc}")
                metrics = {"erro": str(exc)}

        metrics["demo_dim"] = demo_dim
        metrics["n_epochs"] = n_epochs
        results[config_name] = metrics

        acc = metrics.get("accuracy", "n/a")
        f1  = metrics.get("macro_f1",  "n/a")
        logger.info(f"  {config_name}: Acc={acc} | F1={f1}")

    a = results.get("config_A_sem_demo",   {})
    b = results.get("config_B_late_fusion", {})
    if "accuracy" in a and "accuracy" in b:
        delta_acc = round(b["accuracy"] - a["accuracy"], 4)
        delta_f1  = round(b["macro_f1"] - a["macro_f1"], 4) if "macro_f1" in a and "macro_f1" in b else None
        results["delta_B_minus_A"] = {"accuracy": delta_acc, "macro_f1": delta_f1}
        logger.info(f"\n[Ablação] Δ (B − A): Acc={delta_acc:+.4f}"
                    + (f" | F1={delta_f1:+.4f}" if delta_f1 is not None else ""))

    logger.info("\n" + "=" * 70)
    logger.info("RESULTADO ABLAÇÃO — Late Fusion Demográfica")
    logger.info("=" * 70)
    logger.info(f"{'Config':<35} {'Accuracy':>8} {'F1 Macro':>8}")
    logger.info("-" * 55)
    for key in ["config_A_sem_demo", "config_B_late_fusion"]:
        m = results.get(key, {})
        logger.info(
            f"{key:<35} "
            f"{m.get('accuracy', 'n/a'):>8} "
            f"{m.get('macro_f1', 'n/a'):>8}"
        )
    if "delta_B_minus_A" in results:
        d = results["delta_B_minus_A"]
        logger.info(
            f"{'Δ (B − A)':<35} "
            f"{d.get('accuracy', 'n/a'):>+8} "
            + (f"{d.get('macro_f1', 'n/a'):>+8}" if d.get('macro_f1') is not None else "     n/a")
        )
    logger.info("=" * 70)

    results["meta"] = {
        "fonte_demo":    "real_fapesp" if use_real_demo else "sintetico_correlacionado",
        "n_epochs":      n_epochs,
        "random_seed":   random_seed,
        "demo_features": ["age_norm (birth_year / ref_year=2021)", "sex_binary (M=1, F=0)"],
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE ORQUESTRADORA
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedTraining:
    """
    Encapsula o pipeline MOSAIC-FL completo: carregamento, FL, RAG, baseline e ablation.

    Usada pelos dois orquestradores:
      - run_training.py          → instancia com db_url real (dados FAPESP)
      - run_experiments_simulation.py → instancia sem db_url (dados sintéticos)
    """

    def __init__(
        self,
        log_file: str,
        db_url: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        self.log_file = log_file
        self.db_url = db_url
        self.data_source = data_source

        self.client_loaders: Optional[Dict] = None
        self.test_loader: Optional[DataLoader] = None
        self.vocab_map: Optional[Dict] = None
        self.total: int = 0
        self.demographics_by_client: Optional[Dict] = None
        self.test_loader_demo: Optional[DataLoader] = None
        self.cal_loader: Optional[DataLoader] = None
        self.history: Optional[Dict] = None
        self.global_model: Optional[SimplifiedBEHRT] = None
        self._last_round: int = 0

        self._metrics_store = get_metrics_store(db_url)

    def load_from_db(self, db_url: str) -> None:
        (
            self.client_loaders,
            self.test_loader,
            self.vocab_map,
            self.total,
            self.demographics_by_client,
            self.test_loader_demo,
            self.cal_loader,
        ) = prepare_dataloaders_from_db(db_url)

    def load_synthetic(self) -> None:
        df_raw = load_with_fallback(allow_synthetic=True)
        preprocessor = EHRPreprocessor()
        self.client_loaders, self.test_loader, self.vocab_map, self.total = \
            prepare_dataloaders(df_raw, preprocessor)

    def train(self) -> None:
        self.history, self.global_model = run_federated_learning(
            self.client_loaders,
            self.test_loader,
            self.total,
            vocab=self.vocab_map,
            cal_loader=self.cal_loader,
        )
        if self.history and "rounds" in self.history and self.history["rounds"]:
            self._last_round = self.history["rounds"][-1]
            self._metrics_store.save(
                round_num=self._last_round,
                metrics={
                    "accuracy":      self.history["accuracy"][-1] if self.history.get("accuracy") else None,
                    "loss":          self.history["loss"][-1] if self.history.get("loss") else None,
                    "macro_auc":     None,
                    "macro_f1":      None,
                    "ece":           None,
                    "per_class_auc": None,
                    "per_class_f1":  None,
                },
                data_source=self.data_source,
            )

    def run_rag(self) -> Dict:
        result = run_rag_pipeline(self.global_model, self.vocab_map, self.test_loader)
        p_metrics = result.get("precision_metrics", {})
        if p_metrics:
            k = p_metrics.get("k", 3)
            self._metrics_store.save(
                round_num=self._last_round,
                metrics={
                    "rag_precision_at_k":      p_metrics.get(f"precision_at_{k}"),
                    "rag_k":                   k,
                    "rag_per_class_precision": p_metrics.get(f"per_class_precision_at_{k}"),
                },
                data_source=self.data_source,
            )
        return result

    def run_baseline(self) -> Dict:
        result = run_baseline_rf(
            self.client_loaders,
            self.test_loader,
            class_labels=list(MODEL_CFG.class_labels),
        )
        baseline_path = Path("experiments/data") / f"baseline_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Baseline salvo: {baseline_path}")
        m_a = result.get("opcao_a_centralizado") or {}
        if m_a:
            self._metrics_store.save(
                round_num=0,
                metrics={
                    "accuracy":  m_a.get("accuracy"),
                    "macro_auc": m_a.get("macro_auc"),
                    "macro_f1":  m_a.get("macro_f1"),
                    "ece":       m_a.get("ece"),
                },
                data_source=f"{self.data_source}_baseline_rf",
            )
        return result

    def run_ablation(self) -> Dict:
        result = run_ablation_demographics(
            client_loaders=self.client_loaders,
            test_loader=self.test_loader,
            demographics_by_client=self.demographics_by_client,
            test_loader_demo=self.test_loader_demo,
            n_epochs=10,
        )
        ablation_path = Path("experiments/data") / f"ablation_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Ablation salvo: {ablation_path}")
        return result

    def summarize(self, rag_result: Dict, baseline_result: Dict, ablation_result: Dict) -> None:
        logger.info("=" * 60)
        logger.info("CONCLUÍDO")
        logger.info(f"  Modo dados:     {self.data_source}")
        logger.info(f"  Clientes FL:    {len(self.client_loaders)}")
        logger.info(f"  RAG confiável:  {rag_result.get('confiavel', False)}")
        m_a = baseline_result.get("opcao_a_centralizado") or {}
        if m_a:
            logger.info(
                f"  RF centralizado: Acc={m_a.get('accuracy','?')}  "
                f"AUC={m_a.get('macro_auc','?')}  F1={m_a.get('macro_f1','?')}"
            )
        delta = ablation_result.get("delta_B_minus_A", {})
        if delta:
            logger.info(f"  Ablation Δ Acc: {delta.get('accuracy', 'n/a'):+}")
        logger.info(f"  Logs em:        {self.log_file}")
        logger.info("=" * 60)

        logger.info(
            "TREINAMENTO_COMPLETO status=ok fl_rounds=%d rag_ok=%s baseline_rf_ok=%s ablation_ok=%s log=%s",
            self._last_round,
            not bool(rag_result.get("erro")),
            not bool(baseline_result.get("erro")),
            not bool(ablation_result.get("erro")),
            self.log_file,
        )
