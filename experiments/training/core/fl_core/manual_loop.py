"""manual_loop.py — Loop de aprendizado federado manual (sem Ray) — sequencial, leve, didático."""
import json
import logging
import random
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from infrastructure.shared.checkpoint_store import CheckpointStore, get_checkpoint_store
from mosaicfl.core.client import FedProxClient
from mosaicfl.core.config import (
    CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
    DEVICE, FED_CFG, FL_DB_URL, MIN_ROUNDS, MODEL_CFG, NUM_ROUNDS,
    DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
)
from mosaicfl.core.model import SimplifiedBEHRT

from .aggregation import aggregate_fedavg, aggregate_fednova, apply_dp_noise
from .evaluation import evaluate_global_model

logger = logging.getLogger(__name__)


def _sample_gpu_power_w() -> Optional[float]:
    """Amostra a potência instantânea da GPU (Watts) via nvidia-smi.

    Retorna None em qualquer falha (sem GPU NVIDIA, driver ausente, timeout) —
    nunca interrompe o treinamento por causa de coleta de métrica de energia.
    Relevante para viabilidade de implantação em ambientes com energia/água
    limitadas para resfriamento — custo energético real, não estimado.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _evaluate_subgroups(
    model: SimplifiedBEHRT,
    test_loader_origin: DataLoader,
    device,
    origin_labels: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Avalia accuracy/F1 macro por origem hospitalar — uma única passagem sobre o
    checkpoint final já restaurado (não é repetido por rodada). Usado para o
    contraste do Experimento 3 (fase 5, iid_simulado) e, de brinde, também
    disponível na fase 3 (non-IID natural), já que o mecanismo é o mesmo nos
    dois modos (ver dataloaders.py::prepare_dataloaders_from_db)."""
    from sklearn.metrics import f1_score

    model.eval()
    all_preds, all_labels, all_origins = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_dia, batch_origin in test_loader_origin:
            batch_x, batch_dia = batch_x.to(device), batch_dia.to(device)
            logits = model(batch_x, dia_relativo=batch_dia)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(batch_y)
            all_origins.append(batch_origin)

    preds   = torch.cat(all_preds).numpy()
    labels  = torch.cat(all_labels).numpy()
    origins = torch.cat(all_origins).numpy()

    result: Dict[str, Dict[str, float]] = {}
    for origin_id in sorted(set(origins.tolist())):
        mask = origins == origin_id
        n = int(mask.sum())
        if n == 0:
            continue
        label = (origin_labels or {}).get(origin_id, str(origin_id))
        result[label] = {
            "n":        n,
            "accuracy": float((preds[mask] == labels[mask]).mean()),
            "f1_macro": float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0)),
        }
    return result


def run_federated_learning_manual(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
    cal_loader: DataLoader = None,
    test_loader_origin: Optional[DataLoader] = None,
    origin_labels: Optional[Dict[int, str]] = None,
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
    dp_epsilon_simple: Optional[float] = None  # None quando DP desabilitado (FL_DP_NOISE=0)
    checkpoint_store = override_checkpoint_store if override_checkpoint_store is not None else get_checkpoint_store(FL_DB_URL)

    # RDP (Rényi DP) via opacus — cota mais apertada que a composição simples acima,
    # para o mesmo ruído aplicado (McMahan et al. 2018 continua sendo o mecanismo;
    # isto só troca a contabilidade/prova de privacidade). sample_rate=1.0 porque
    # os 2 clientes participam de toda rodada — sem subamostragem, sem amplificação.
    _rdp_accountant = None
    if DP_NOISE_MULTIPLIER > 0:
        from opacus.accountants import RDPAccountant
        _rdp_accountant = RDPAccountant()

    # Monitoramento de recursos computacionais (psutil)
    _proc = psutil.Process()
    _proc.cpu_percent(interval=None)  # primeira chamada calibra — descartada; próximas retornam % real
    _peak_ram_mb  = 0.0
    _cpu_samples: List[float] = []
    _gpu_power_samples: List[float] = []  # fica vazio se não houver GPU NVIDIA — normal em treino CPU-only

    # Registra o treinamento antes do loop — garante 1 checkpoint por treinamento
    training_id: Optional[int] = None
    import os
    partition_mode = os.environ.get("FL_PARTITION_MODE", "natural").strip().lower()
    if not sensitivity_mode:
        log_file = os.environ.get("FL_LOG_FILE", "")
        training_id = checkpoint_store.register_training(
            algorithm=agregador,
            log_file=log_file,
            n_rounds_max=n_rounds,
            checkpoint_criterion=FED_CFG.checkpoint_criterion,
            partition_mode=partition_mode,
        )
        logger.info("training_registered id=%d algorithm=%s criterion=%s n_rounds_max=%d partition_mode=%s",
                    training_id, agregador, FED_CFG.checkpoint_criterion, n_rounds, partition_mode)

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
                {k: torch.tensor(v, device=DEVICE) for k, v in zip(global_state.keys(), fit_params)}
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
            dp_epsilon_simple = apply_dp_noise(
                global_state, round_num, len(client_loaders), DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM
            )
            _rdp_accountant.step(noise_multiplier=DP_NOISE_MULTIPLIER, sample_rate=1.0)

        global_model.load_state_dict(global_state)

        loss_global, acc_global, f1_global, per_class_f1 = evaluate_global_model(global_model, test_loader)
        round_duration = time.time() - round_start
        model_size_mb = sum(p.numel() * 4 for p in global_model.parameters()) / (1024 ** 2)
        comm_mb = model_size_mb * len(client_states) * 2

        _ram_mb  = _proc.memory_info().rss / (1024 ** 2)
        _cpu_pct = _proc.cpu_percent(interval=None)
        _peak_ram_mb = max(_peak_ram_mb, _ram_mb)
        _cpu_samples.append(_cpu_pct)
        _gpu_power_w = _sample_gpu_power_w()
        if _gpu_power_w is not None:
            _gpu_power_samples.append(_gpu_power_w)

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

    # Energia estimada por amostragem (potência média × duração) — não é medição
    # contínua, é uma estimativa best-effort. None quando não há GPU NVIDIA (CPU-only).
    _gpu_avg_power_w:  Optional[float] = None
    _gpu_peak_power_w: Optional[float] = None
    _gpu_energy_wh:    Optional[float] = None
    if _gpu_power_samples:
        _gpu_avg_power_w  = sum(_gpu_power_samples) / len(_gpu_power_samples)
        _gpu_peak_power_w = max(_gpu_power_samples)
        _gpu_energy_wh    = _gpu_avg_power_w * (overall_duration / 3600)

    logger.info(
        "resource_summary duration=%.1fs peak_ram=%.0fMB avg_cpu=%.1f%% "
        "gpu_avg_power=%sW gpu_peak_power=%sW gpu_energy=%sWh rounds=%d",
        overall_duration, _peak_ram_mb, _avg_cpu,
        f"{_gpu_avg_power_w:.1f}" if _gpu_avg_power_w is not None else "n/a",
        f"{_gpu_peak_power_w:.1f}" if _gpu_peak_power_w is not None else "n/a",
        f"{_gpu_energy_wh:.4f}" if _gpu_energy_wh is not None else "n/a",
        round_num,
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
            gpu_avg_power_w=_gpu_avg_power_w,
            gpu_peak_power_w=_gpu_peak_power_w,
            gpu_energy_wh=_gpu_energy_wh,
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

    # Avaliação por subgrupo de origem hospitalar — uma única passagem sobre o
    # checkpoint final (não por rodada). Disponível nos dois modos de partição
    # (natural e iid_simulado); é o que sustenta o contraste causal do
    # Experimento 3 quando este treino é a fase 5 do pipeline.
    subgroup_metrics: Optional[Dict] = None
    if test_loader_origin is not None:
        try:
            subgroup_metrics = _evaluate_subgroups(global_model, test_loader_origin, DEVICE, origin_labels)
            logger.info("subgroup_metrics partition_mode=%s %s", partition_mode, subgroup_metrics)
        except Exception as exc:
            logger.warning("Falha ao avaliar por subgrupo de origem: %s", exc)

    # ε via RDP (opacus) — mesmo mecanismo de ruído, cota mais apertada que a
    # composição simples. delta=1e-5 igual ao default de apply_dp_noise, para
    # os dois números serem diretamente comparáveis.
    _DP_DELTA = 1e-5
    dp_epsilon_rdp: Optional[float] = None
    if _rdp_accountant is not None:
        dp_epsilon_rdp = _rdp_accountant.get_epsilon(delta=_DP_DELTA)

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
        "partition_mode":   partition_mode,
        "subgroup_metrics": subgroup_metrics,
        "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
        "dp_max_grad_norm":    DP_MAX_GRAD_NORM,
        "dp_epsilon_simple":   dp_epsilon_simple,  # None se DP desabilitado — composição gaussiana simples
        "dp_epsilon_rdp":      dp_epsilon_rdp,      # None se DP desabilitado — Rényi DP (opacus), cota mais apertada
    }
    if dp_epsilon_simple is not None:
        logger.info(
            "dp_summary sigma=%.2f clip=%.2f rounds=%d delta=%.0e "
            "epsilon_simple=%.3f epsilon_rdp=%.3f (rdp = cota mais apertada, mesmo ruído)",
            DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM, round_num, _DP_DELTA,
            dp_epsilon_simple, dp_epsilon_rdp,
        )

    # Preferência: pós-calibração (report_cal) > pré-calibração (report_raw) > None.
    # Fica disponível em history (→ metrics_store, via orchestrator) e em fl_trainings
    # (→ update_evaluation_metrics, abaixo) — consultável por SQL sem re-parsing de log/JSONB.
    _best_report = report_cal or report_raw
    history["macro_auc"] = _best_report.macro_auc if _best_report else None
    history["macro_f1_report"] = _best_report.macro_f1 if _best_report else None
    history["ece"] = _best_report.calibration.ece if _best_report else None
    # ece_pre é sempre a saída bruta do modelo (report_raw), independente de report_cal
    # existir ou não — não segue a preferência pós>pré usada nos outros campos, porque
    # aqui o objetivo é exatamente o par antes/depois lado a lado.
    history["ece_pre"] = report_raw.calibration.ece if report_raw else None

    history["dp_epsilon_simple"] = dp_epsilon_simple
    history["dp_epsilon_rdp"]    = dp_epsilon_rdp

    if training_id is not None:
        try:
            checkpoint_store.update_evaluation_metrics(
                training_id=training_id,
                macro_auc=history["macro_auc"],
                macro_f1=history["macro_f1_report"],
                ece=history["ece"],
                ece_pre=history["ece_pre"],
                dp_noise_multiplier=DP_NOISE_MULTIPLIER if dp_epsilon_simple is not None else None,
                dp_max_grad_norm=DP_MAX_GRAD_NORM if dp_epsilon_simple is not None else None,
                dp_epsilon_simple=dp_epsilon_simple,
                dp_epsilon_rdp=dp_epsilon_rdp,
            )
        except Exception as exc:
            logger.warning("Falha ao salvar macro_auc/macro_f1/ece/ece_pre/dp_* em fl_trainings: %s", exc)

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
