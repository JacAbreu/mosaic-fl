"""
Núcleo do aprendizado federado: agregação, avaliação global e loops de treinamento.

  aggregate_fedavg              — média ponderada de state_dicts (FedAvg)
  aggregate_fednova             — média ponderada normalizada por passos efetivos τ_i (FedNova)
  evaluate_global_model         — acurácia e loss no conjunto de teste
  run_federated_learning_manual — FL sequencial sem Ray
  run_federated_learning_ray    — FL paralelo com Ray/Flower simulation
  run_federated_learning        — roteador manual↔Ray baseado em USE_RAY

Algoritmo de agregação selecionado por FED_CFG.use_fednova (config.py).
"""
import json
import logging
import math
import random
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from infrastructure.shared.checkpoint_store import CheckpointStore, get_checkpoint_store
from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import (
    BATCH_SIZE, CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
    DEVICE, FED_CFG, FL_DB_URL, LOCAL_EPOCHS, MIN_ROUNDS, MODEL_CFG, NUM_ROUNDS, USE_RAY,
    DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
)
from mosaicfl.core.model import SimplifiedBEHRT

logger = logging.getLogger(__name__)


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


def aggregate_fednova(
    global_state: OrderedDict,
    client_states: List[OrderedDict],
    weights: List[int],
    tau_values: List[int],
) -> Tuple[OrderedDict, float]:
    """Agrega state_dicts via FedNova — normaliza updates por passos efetivos τ_i.

    Corrige o viés de agregação em clientes heterogêneos: clientes com mais dados
    (mais batches por rodada) têm seus updates normalizados por τ_i antes de agregar,
    equalizando a contribuição independente do volume local.

    Fórmula (Wang et al. 2020):
        τ_eff = Σ p_i · τ_i
        w_{t+1} = w_t + τ_eff · Σ p_i · (w_i − w_t) / τ_i

    Referência: Wang et al. 2020 — "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization"
    """
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Peso total de agregação é zero.")

    tau_eff = sum(n * tau for n, tau in zip(weights, tau_values)) / total_weight

    new_state = OrderedDict()
    for key in global_state.keys():
        w_t = global_state[key].float()
        normalized_delta = torch.zeros_like(w_t)
        for state, n, tau in zip(client_states, weights, tau_values):
            p_i = n / total_weight
            normalized_delta += p_i * (state[key].float() - w_t) / max(tau, 1)
        new_state[key] = (w_t + tau_eff * normalized_delta).to(global_state[key].dtype)

    return new_state, tau_eff


def apply_dp_noise(
    global_state: OrderedDict,
    round_num: int,
    n_clients: int,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float = 1e-5,
) -> float:
    """Adiciona ruído gaussiano ao estado global agregado (DP-FedAvg, McMahan et al. 2018).

    noise_std = σ × S / n_clients
    ε por rodada ≈ √(2 ln(1.25/δ)) / σ   (mecanismo Gaussiano — cota superior)
    Para cotas mais apertadas, usar RDP/moments accountant (ex: Opacus).

    Retorna ε acumulado (composição simples × rodadas).
    """
    noise_std = noise_multiplier * max_grad_norm / max(n_clients, 1)
    with torch.no_grad():
        for key in global_state:
            noise = torch.normal(0.0, noise_std, size=global_state[key].shape)
            global_state[key] = (global_state[key].float() + noise).to(global_state[key].dtype)

    eps_per_round = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    eps_accumulated = eps_per_round * round_num
    logger.info(
        "dp_noise σ=%.2f S=%.2f noise_std=%.6f n=%d | "
        "ε_rodada≈%.3f ε_acum≈%.3f δ=%.0e (cota superior — composição simples)",
        noise_multiplier, max_grad_norm, noise_std, n_clients,
        eps_per_round, eps_accumulated, delta,
    )
    return eps_accumulated


def evaluate_global_model(
    model: SimplifiedBEHRT, test_loader: DataLoader
) -> Tuple[float, float, float, List[float]]:
    """Avalia modelo global no conjunto de teste.

    Retorna (loss, accuracy, f1_macro, per_class_f1).
    f1_macro é o critério primário de seleção do checkpoint — mais robusto que
    accuracy em datasets desbalanceados (zero_division=0 penaliza classes não previstas).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y, batch_dia in test_loader:
            batch_x   = batch_x.to(DEVICE)
            batch_y   = batch_y.to(DEVICE)
            batch_dia = batch_dia.to(DEVICE)
            logits    = model(batch_x, dia_relativo=batch_dia)
            loss      = criterion(logits, batch_y)
            loss_sum += loss.item() * batch_y.size(0)
            _, predicted = torch.max(logits, dim=1)
            total   += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    avg_loss     = loss_sum / total if total > 0 else 0.0
    accuracy     = correct  / total if total > 0 else 0.0
    f1_macro     = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    return avg_loss, accuracy, f1_macro, per_class_f1


def run_federated_learning_manual(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
    cal_loader: DataLoader = None,
    # Overrides para análise de sensibilidade — não usar no treinamento principal
    override_num_rounds: Optional[int] = None,
    override_use_fednova: Optional[bool] = None,
    override_random_seed: Optional[int] = None,
    override_checkpoint_store: Optional[CheckpointStore] = None,
    sensitivity_mode: bool = False,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL manualmente, sem Ray — sequencial, leve, didático.

    Os parâmetros override_* permitem rodar análises de sensibilidade (ex: múltiplas seeds,
    rodadas reduzidas) sem alterar o FedConfig global. sensitivity_mode=True suprime
    calibração, avaliação detalhada e escrita de arquivos — útil para loops de benchmark.
    """
    n_rounds    = override_num_rounds    if override_num_rounds    is not None else NUM_ROUNDS
    use_fednova = override_use_fednova   if override_use_fednova   is not None else FED_CFG.use_fednova
    seed        = override_random_seed   if override_random_seed   is not None else FED_CFG.random_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO MANUAL (SEM RAY)")
    logger.info("=" * 60)
    agregador = "FedNova" if use_fednova else "FedAvg"
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {n_rounds} | Agregação: {agregador} | seed={seed}")

    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    global_state = global_model.state_dict()

    history = {
        "rounds": [], "accuracy": [], "loss": [],
        "communication_mb": [], "client_losses": [],
        "tau_eff": [], "f1_macro": [], "per_class_f1": [],
        "round_duration_s": [],
    }

    stable_count       = 0
    prev_criterion_val = 0.0
    converged_round    = None
    best_criterion_val = 0.0  # melhor valor do critério selecionado (f1_macro ou accuracy)
    best_f1            = 0.0  # f1_macro na rodada do melhor checkpoint (sempre rastreado)
    best_accuracy      = 0.0  # accuracy na rodada do melhor checkpoint (sempre rastreado)
    best_round         = 0
    checkpoint_store = override_checkpoint_store if override_checkpoint_store is not None else get_checkpoint_store(FL_DB_URL)

    # Monitoramento de recursos computacionais (psutil)
    _proc = psutil.Process()
    _proc.cpu_percent(interval=None)  # primeira chamada calibra — descartada; próximas retornam % real
    _peak_ram_mb  = 0.0
    _cpu_samples: List[float] = []

    # Registra o treinamento antes do loop — garante 1 checkpoint por treinamento
    training_id: Optional[int] = None
    if not sensitivity_mode:
        import os
        log_file = os.environ.get("FL_LOG_FILE", "")
        training_id = checkpoint_store.register_training(
            algorithm=agregador,
            log_file=log_file,
            n_rounds_max=n_rounds,
            checkpoint_criterion=FED_CFG.checkpoint_criterion,
        )
        logger.info("training_registered id=%d algorithm=%s criterion=%s n_rounds_max=%d",
                    training_id, agregador, FED_CFG.checkpoint_criterion, n_rounds)

    overall_start = time.time()

    for round_num in range(1, n_rounds + 1):
        round_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"RODADA {round_num}/{n_rounds}")
        logger.info(f"{'='*60}")

        client_states, client_weights, client_metrics = [], [], []

        for cid, (train_loader, val_loader) in client_loaders.items():
            logger.info(f"  [Cliente {cid}] Treinando...")
            client = FedProxClient(cid, train_loader, val_loader)
            global_params = [v.cpu().numpy() for v in global_state.values()]
            client.set_parameters(global_params)
            fit_params, num_samples, metrics = client.fit(
                global_params, config={"current_round": round_num}
            )
            fit_state = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(global_state.keys(), fit_params)}
            )
            client_states.append(fit_state)
            client_weights.append(num_samples)
            client_metrics.append(metrics)
            tau_i     = metrics.get("tau", 0)
            grad_norm = metrics.get("grad_norm", 0.0)
            logger.info(
                f"  [Cliente {cid}] Loss: {metrics['loss']:.4f} | Amostras: {num_samples} "
                f"| τ={tau_i} | grad_norm={grad_norm:.4f}"
            )

        logger.info(f"  [Servidor] Agregando pesos de {len(client_states)} clientes...")
        if use_fednova:
            tau_values = [m.get("tau", 1) for m in client_metrics]
            global_state, tau_eff = aggregate_fednova(global_state, client_states, client_weights, tau_values)
            logger.info(f"  [FedNova] τ={tau_values} | τ_eff={tau_eff:.1f}")
        else:
            global_state = aggregate_fedavg(client_states, client_weights)
            tau_eff = None

        if DP_NOISE_MULTIPLIER > 0:
            apply_dp_noise(global_state, round_num, len(client_loaders), DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM)

        global_model.load_state_dict(global_state)

        loss_global, acc_global, f1_global, per_class_f1 = evaluate_global_model(global_model, test_loader)
        round_duration = time.time() - round_start
        model_size_mb = sum(p.numel() * 4 for p in global_model.parameters()) / (1024 ** 2)
        comm_mb = model_size_mb * len(client_states) * 2

        _ram_mb  = _proc.memory_info().rss / (1024 ** 2)
        _cpu_pct = _proc.cpu_percent(interval=None)
        _peak_ram_mb = max(_peak_ram_mb, _ram_mb)
        _cpu_samples.append(_cpu_pct)

        logger.info(
            f"  [Servidor] Rodada {round_num} em {round_duration:.2f}s | "
            f"Loss: {loss_global:.4f} | Acc: {acc_global:.4f} | F1-macro: {f1_global:.4f}"
        )
        logger.info(
            "  [Recursos] RAM=%.0fMB (pico=%.0fMB) CPU=%.1f%% Rodada=%.2fs",
            _ram_mb, _peak_ram_mb, _cpu_pct, round_duration,
        )

        history["rounds"].append(round_num)
        history["accuracy"].append(acc_global)
        history["loss"].append(loss_global)
        history["communication_mb"].append(comm_mb)
        history["client_losses"].append([m["loss"] for m in client_metrics])
        history["tau_eff"].append(tau_eff)
        history["f1_macro"].append(f1_global)
        history["per_class_f1"].append(per_class_f1)
        history["round_duration_s"].append(round_duration)

        criterion_value = f1_global if FED_CFG.checkpoint_criterion == "f1_macro" else acc_global
        if criterion_value > best_criterion_val:
            best_criterion_val = criterion_value
            best_f1            = f1_global
            best_accuracy      = acc_global
            best_round         = round_num
            checkpoint_store.save(
                round_num=round_num,
                state_dict=global_model.state_dict(),
                vocab=vocab or {},
                accuracy=acc_global,
                loss=loss_global,
                training_id=training_id,
            )
            logger.info(
                f"  [Best] Novo melhor checkpoint — rodada {round_num} "
                f"F1-macro={f1_global:.4f} Acc={acc_global:.4f} (training_id={training_id})"
            )

        if round_num > 1:
            delta = abs(criterion_value - prev_criterion_val)
            crit_label = "F1" if FED_CFG.checkpoint_criterion == "f1_macro" else "Acc"
            if round_num <= MIN_ROUNDS:
                logger.info(f"  [Warm-up {round_num}/{MIN_ROUNDS}] Δ {crit_label}={delta:.5f} — convergência suspensa até rodada {MIN_ROUNDS}")
            elif delta < CONVERGENCE_THRESHOLD:
                stable_count += 1
                logger.info(f"  [Convergência] Δ {crit_label}={delta:.5f} < {CONVERGENCE_THRESHOLD} ({stable_count}/{CONVERGENCE_PATIENCE})")
            else:
                stable_count = 0

            if stable_count >= CONVERGENCE_PATIENCE and converged_round is None:
                converged_round = round_num
                logger.info(f"\nCONVERGENCIA ATINGIDA na rodada {round_num}!")
                break

        prev_criterion_val = criterion_value

    overall_duration = time.time() - overall_start
    _avg_cpu = sum(_cpu_samples) / len(_cpu_samples) if _cpu_samples else 0.0

    logger.info(
        "resource_summary duration=%.1fs peak_ram=%.0fMB avg_cpu=%.1f%% rounds=%d",
        overall_duration, _peak_ram_mb, _avg_cpu, round_num,
    )

    if best_round == 0:
        # Nenhuma rodada superou best_criterion_val=0.0 — usa a última rodada como referência.
        best_round         = round_num
        best_f1            = history["f1_macro"][-1]
        best_accuracy      = history["accuracy"][-1]
        best_criterion_val = history["f1_macro"][-1] if FED_CFG.checkpoint_criterion == "f1_macro" else history["accuracy"][-1]

    if training_id is not None:
        checkpoint_store.complete_training(
            training_id=training_id,
            n_rounds_done=round_num,
            best_round=best_round,
            best_accuracy=best_accuracy,
            converged=bool(converged_round),
            total_duration_s=overall_duration,
            peak_ram_mb=_peak_ram_mb,
            avg_cpu_pct=_avg_cpu,
        )

    best_ckpt = checkpoint_store.load_best(training_id=training_id)
    if best_ckpt is not None:
        global_model.load_state_dict(best_ckpt["model_state"])
        loaded_round = best_ckpt.get("checkpoint_round", best_round)
        logger.info(
            f"  [Best] Modelo restaurado da rodada {loaded_round} "
            f"F1-macro={best_f1:.4f} Acc={best_accuracy:.4f} training_id={training_id}"
        )
    else:
        logger.warning("  [Best] load_best retornou None — usando modelo da última rodada")

    logger.info(f"\n{'='*60}")
    logger.info("SIMULAÇÃO CONCLUÍDA")
    logger.info(f"{'='*60}")
    logger.info(f"  Rodadas:       {round_num}")
    logger.info(f"  Convergência:  {'Sim (rodada ' + str(converged_round) + ')' if converged_round else 'Não'}")
    logger.info(f"  Melhor rodada: {best_round} (F1-macro={best_f1:.4f} | Acc={best_accuracy:.4f})")
    logger.info(f"  F1-macro final (última rodada): {history['f1_macro'][-1]:.4f}")
    logger.info(f"  Acurácia final (última rodada): {history['accuracy'][-1]:.4f}")
    logger.info(f"  Loss:          {history['loss'][-1]:.4f}")
    logger.info(f"  Tráfego:       {sum(history['communication_mb']):.2f} MB")
    logger.info(f"  Tempo total:   {overall_duration:.2f}s ({overall_duration/60:.2f} min)")
    logger.info(f"{'='*60}")

    logger.info(
        "FL_TRAINING_COMPLETE rounds=%d converged=%s best_round=%d "
        "best_f1_macro=%.4f best_accuracy=%.4f "
        "last_f1_macro=%.4f last_accuracy=%.4f loss=%.4f duration_s=%.1f traffic_mb=%.2f",
        round_num, bool(converged_round),
        best_round, best_f1, best_accuracy,
        history["f1_macro"][-1], history["accuracy"][-1], history["loss"][-1],
        overall_duration, sum(history["communication_mb"]),
    )

    if sensitivity_mode:
        logger.info("[sensitivity_mode] Calibração e avaliação detalhada suprimidas.")
        return history, global_model

    hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Histórico salvo: {hist_path}")

    from mosaicfl.core.calibration import IsotonicCalibrator, TemperatureScaler
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

    # ── Temperature scaling ───────────────────────────────────────────────────
    scaler = TemperatureScaler()
    try:
        scaler.fit(global_model, _calib_loader, device=str(DEVICE))
        n_cal = len(_calib_loader.dataset) if hasattr(_calib_loader, "dataset") else "?"
        logger.info(f"temperature_scaling T={scaler.T:.4f} (cal_set={n_cal} amostras)")
    except Exception as exc:
        logger.warning(f"temperature_scaling_failed ({exc}) — T mantido em 1.0")

    try:
        report_cal = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=scaler.T)
        logger.info(f"Avaliação pós-temperature — ECE={report_cal.calibration.ece:.4f} "
                    f"AUC={report_cal.macro_auc:.4f} F1={report_cal.macro_f1:.4f}")
        print_report(report_cal)
    except Exception as exc:
        logger.warning(f"Avaliação pós-temperature falhou: {exc}")
        report_cal = None

    # ── Calibração Isotônica OvR ─────────────────────────────────────────────
    iso = IsotonicCalibrator()
    ece_iso: Optional[float] = None
    try:
        iso.fit(global_model, _calib_loader, device=str(DEVICE), num_classes=MODEL_CFG.num_classes)
        # ECE com calibração isotônica (avaliado no test_loader para comparação justa)
        all_logits, all_labels_t = [], []
        global_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                bx, by, bdia = batch[0].to(DEVICE), batch[1], batch[2].to(DEVICE)
                all_logits.append(global_model(bx, dia_relativo=bdia).cpu())
                all_labels_t.append(by)
        all_logits   = torch.cat(all_logits)
        all_labels_t = torch.cat(all_labels_t)
        ece_iso = iso.compute_ece(all_logits, all_labels_t)
        ece_temp = report_cal.calibration.ece if report_cal else None
        logger.info(
            "calibration_comparison ECE_pre=%.4f ECE_temperature=%.4f ECE_isotonic=%.4f — melhor=%s",
            report_raw.calibration.ece if report_raw else float("nan"),
            ece_temp if ece_temp is not None else float("nan"),
            ece_iso,
            "isotonic" if (ece_temp is not None and ece_iso < ece_temp) else "temperature",
        )
    except Exception as exc:
        logger.warning(f"isotonic_calibration_failed: {exc}")

    import dataclasses
    evaluation_payload = {
        "best_round":       best_round,
        "total_rounds":     round_num,
        "best_f1_macro":    best_f1,
        "best_accuracy":    best_accuracy,
        "temperature":      round(scaler.T, 4),
        "ece_isotonic":     ece_iso,
        "pre_calibration":  dataclasses.asdict(report_raw) if report_raw  else None,
        "post_calibration": dataclasses.asdict(report_cal) if report_cal  else None,
    }

    # Banco é a fonte da verdade — persiste junto com o checkpoint.
    # O arquivo é mantido como cache de leitura rápida (sobrescrito a cada run).
    checkpoint_store.save(
        round_num=best_round,
        state_dict=global_model.state_dict(),
        vocab=vocab or {},
        accuracy=best_accuracy,
        loss=history["loss"][best_round - 1],
        temperature=scaler.T,
        training_id=training_id,
        evaluation_json=evaluation_payload,
    )
    logger.info(
        f"checkpoint_saved_postgres round={best_round} accuracy={best_accuracy:.4f} "
        f"T={scaler.T:.4f} vocab_size={len(vocab or {})} evaluation_json=saved"
    )

    eval_path = Path("experiments/logs") / f"evaluation_best_r{best_round}_of_{round_num}.json"
    try:
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(evaluation_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Relatório de avaliação (cache local): {eval_path}")
    except Exception as exc:
        logger.warning(f"Falha ao salvar cache local de avaliação: {exc}")

    if training_id is not None:
        try:
            checkpoint_store.save_round_history(
                training_id=training_id,
                rounds=history.get("rounds", []),
                accuracies=history.get("accuracy", []),
                losses=history.get("loss", []),
                tau_effs=history.get("tau_eff"),
                f1_macros=history.get("f1_macro"),
                per_class_f1s=history.get("per_class_f1"),
                round_durations=history.get("round_duration_s"),
            )
        except Exception as exc:
            logger.warning("Falha ao salvar histórico por rodada: %s", exc)

    history["training_id"] = training_id
    return history, global_model


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

    from experiments.experiment_server import start_server
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
    if bool(USE_RAY):
        logger.info("Modo Ray ativado (USE_RAY=True).")
        return run_federated_learning_ray(client_loaders, test_loader, total_train_samples, vocab=vocab)
    else:
        logger.info("Modo manual ativado (USE_RAY=False). Ray NÃO é necessário.")
        return run_federated_learning_manual(
            client_loaders, test_loader, total_train_samples, vocab=vocab, cal_loader=cal_loader
        )
