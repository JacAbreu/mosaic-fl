"""
Ablation study de late fusion demográfica e pooled baseline (artefato de pesquisa).

  run_ablation_demographics — ablation_A_sem_demo vs ablation_B_late_fusion
  run_pooled_behrt          — behrt_pooled_A_sem_demo vs behrt_pooled_B_late_fusion
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mosaicfl.core.config import (
    BATCH_SIZE, DEVICE, LOCAL_EPOCHS, LR, MODEL_CFG, NUM_ROUNDS, POOLED_EPOCHS, RANDOM_SEED,
)
from mosaicfl.core.model import SimplifiedBEHRT

logger = logging.getLogger(__name__)


def _make_correlated_demo(labels: torch.Tensor, noise_std: float = 0.15, seed: int = 42) -> torch.Tensor:
    """Gera demográficos sintéticos correlacionados com os labels de prognóstico."""
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
    seeds: Optional[List[int]] = None,
) -> Dict:
    """
    Ablation study: late fusion demográfica (age_norm + sex_binary) no SimplifiedBEHRT.

    ablation_A_sem_demo:    demo_dim=0 (modelo base)
    ablation_B_late_fusion: demo_dim=2 (late fusion com age_norm + sex_binary)

    seeds: lista de seeds para múltiplos runs (recomendado: [42, 7, 123]).
           Resultado reporta média ± desvio-padrão, eliminando sensibilidade à inicialização.
           Padrão: [random_seed] (run único, retrocompatível).
    """
    from mosaicfl.core.evaluation import evaluate, print_report

    random_seed   = random_seed if random_seed is not None else RANDOM_SEED
    seeds         = seeds if seeds is not None else [random_seed]
    use_real_demo = demographics_by_client is not None

    logger.info("=" * 60)
    logger.info("ABLATION — Late Fusion Demográfica")
    logger.info(f"  Fonte dos demográficos: {'dados reais (FAPESP)' if use_real_demo else 'sintéticos correlacionados'}")
    logger.info("  ablation_A_sem_demo:    SimplifiedBEHRT sem demográficos (demo_dim=0)")
    logger.info("  ablation_B_late_fusion: SimplifiedBEHRT + late fusion (demo_dim=2)")
    logger.info(f"  Épocas locais: {n_epochs} | seeds: {seeds}")
    logger.info("=" * 60)

    def _train_local(
        model: nn.Module, loaders: Dict, with_demo: bool,
        demo_loaders: Optional[Dict], seed: int,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        model.train()

        synth_demo_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if with_demo and not use_real_demo:
            for cid, (train_loader, _) in loaders.items():
                all_x, all_y, all_dia = [], [], []
                for bx, by, bdia in train_loader:
                    all_x.append(bx); all_y.append(by); all_dia.append(bdia)
                all_x = torch.cat(all_x); all_y = torch.cat(all_y); all_dia = torch.cat(all_dia)
                synth_demo_cache[cid] = (all_x, all_y, _make_correlated_demo(all_y, seed=seed + cid), all_dia)

        for _ in range(n_epochs):
            if with_demo and use_real_demo and demo_loaders:
                for cid, (demo_train_loader, _) in demo_loaders.items():
                    for batch_x, batch_y, batch_demo, batch_dia in demo_train_loader:
                        batch_x = batch_x.to(DEVICE)
                        batch_y = batch_y.to(DEVICE)
                        batch_demo = batch_demo.to(DEVICE)
                        batch_dia  = batch_dia.to(DEVICE)
                        optimizer.zero_grad()
                        criterion(model(batch_x, demographics=batch_demo, dia_relativo=batch_dia), batch_y).backward()
                        optimizer.step()
            elif with_demo and not use_real_demo:
                for cid, (all_x, all_y, all_demo, all_dia) in synth_demo_cache.items():
                    for bx, by, bd, bdia in DataLoader(TensorDataset(all_x, all_y, all_demo, all_dia),
                                                        batch_size=BATCH_SIZE, shuffle=True):
                        bx, by, bd, bdia = bx.to(DEVICE), by.to(DEVICE), bd.to(DEVICE), bdia.to(DEVICE)
                        optimizer.zero_grad()
                        criterion(model(bx, demographics=bd, dia_relativo=bdia), by).backward()
                        optimizer.step()
            else:
                for cid, (train_loader, _) in loaders.items():
                    for batch_x, batch_y, batch_dia in train_loader:
                        batch_x  = batch_x.to(DEVICE)
                        batch_y  = batch_y.to(DEVICE)
                        batch_dia = batch_dia.to(DEVICE)
                        optimizer.zero_grad()
                        criterion(model(batch_x, dia_relativo=batch_dia), batch_y).backward()
                        optimizer.step()

    def _eval_with_demo(model: nn.Module, t_loader_demo: DataLoader, seed: int) -> Dict:
        model.eval()
        all_preds, all_labels = [], []
        criterion = nn.CrossEntropyLoss()
        total_loss, total_n = 0.0, 0

        with torch.no_grad():
            for batch in t_loader_demo:
                if len(batch) >= 4:
                    bx, by, bd, bdia = batch[0], batch[1], batch[2], batch[3]
                elif len(batch) == 3:
                    bx, by, bd = batch
                    bdia = None
                else:
                    bx, by = batch[0], batch[1]
                    bd   = _make_correlated_demo(by, seed=seed + 99)
                    bdia = None
                bx  = bx.to(DEVICE)
                by  = by.to(DEVICE)
                bd  = bd.to(DEVICE)
                bdia = bdia.to(DEVICE) if bdia is not None else None
                logits = model(bx, demographics=bd, dia_relativo=bdia)
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
        ("ablation_A_sem_demo",    0, False),
        ("ablation_B_late_fusion", 2, True),
    ]:
        logger.info(f"\n[Ablação] Treinando {config_name} (demo_dim={demo_dim}, épocas={n_epochs}, seeds={seeds})...")

        seed_metrics: List[Dict] = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = SimplifiedBEHRT(use_cls_token=True, demo_dim=demo_dim).to(DEVICE)
            _train_local(model, client_loaders, with_demo, demographics_by_client, seed)

            if with_demo:
                t_loader = test_loader_demo if (use_real_demo and test_loader_demo is not None) else test_loader
                m = _eval_with_demo(model, t_loader, seed)
                m["descricao"] = (
                    "SimplifiedBEHRT + late fusion demográfica "
                    f"({'dados reais FAPESP' if use_real_demo else 'sintéticos correlacionados'})"
                )
            else:
                try:
                    report = evaluate(model, test_loader, class_labels=MODEL_CFG.class_labels,
                                      device=str(DEVICE), temperature=1.0)
                    m = {
                        "accuracy":  round(report.accuracy, 4),
                        "macro_f1":  round(report.macro_f1, 4),
                        "macro_auc": round(report.macro_auc, 4) if report.macro_auc else None,
                        "ece":       round(report.calibration.ece, 4),
                        "descricao": "SimplifiedBEHRT sem demográficos (baseline)",
                    }
                except Exception as exc:
                    logger.warning(f"evaluate() falhou para {config_name} seed={seed}: {exc}")
                    m = {"erro": str(exc)}

            m["seed"]     = seed
            m["demo_dim"] = demo_dim
            m["n_epochs"] = n_epochs
            seed_metrics.append(m)
            logger.info(f"    seed={seed}: Acc={m.get('accuracy','n/a')} | F1={m.get('macro_f1','n/a')}")

        valid = [m for m in seed_metrics if "accuracy" in m]
        if valid:
            accs = [m["accuracy"] for m in valid]
            f1s  = [m["macro_f1"] for m in valid if "macro_f1" in m]
            agg: Dict = {
                "accuracy":     round(float(np.mean(accs)), 4),
                "std_accuracy": round(float(np.std(accs)), 4),
                "macro_f1":     round(float(np.mean(f1s)), 4) if f1s else None,
                "std_macro_f1": round(float(np.std(f1s)), 4)  if f1s else None,
                "demo_dim":     demo_dim,
                "n_epochs":     n_epochs,
                "n_seeds":      len(valid),
                "seeds_detail": seed_metrics,
                "descricao":    valid[0].get("descricao", ""),
            }
        else:
            agg = {"erro": "all seeds failed", "seeds_detail": seed_metrics}

        results[config_name] = agg
        acc_str = (f"{agg.get('accuracy','n/a')} ± {agg.get('std_accuracy','n/a')}"
                   if "std_accuracy" in agg else str(agg.get("accuracy", "n/a")))
        f1_str  = (f"{agg.get('macro_f1','n/a')} ± {agg.get('std_macro_f1','n/a')}"
                   if "std_macro_f1" in agg else str(agg.get("macro_f1", "n/a")))
        logger.info(f"  {config_name}: Acc={acc_str} | F1={f1_str}")

    a = results.get("ablation_A_sem_demo",    {})
    b = results.get("ablation_B_late_fusion", {})
    if "accuracy" in a and "accuracy" in b:
        delta_acc = round(b["accuracy"] - a["accuracy"], 4)
        delta_f1  = round(b["macro_f1"] - a["macro_f1"], 4) if "macro_f1" in a and "macro_f1" in b else None
        results["delta_B_minus_A"] = {"accuracy": delta_acc, "macro_f1": delta_f1}
        logger.info(f"\n[Ablação] Δ (B − A): Acc={delta_acc:+.4f}"
                    + (f" | F1={delta_f1:+.4f}" if delta_f1 is not None else ""))

    logger.info("\n" + "=" * 75)
    logger.info("RESULTADO ABLAÇÃO — Late Fusion Demográfica")
    logger.info("=" * 75)
    logger.info(f"{'Config':<35} {'Acc (média±std)':>18} {'F1 (média±std)':>18}")
    logger.info("-" * 75)
    for key in ["ablation_A_sem_demo", "ablation_B_late_fusion"]:
        m = results.get(key, {})
        acc_str = (f"{m.get('accuracy','n/a')} ± {m.get('std_accuracy','n/a')}"
                   if "std_accuracy" in m else str(m.get("accuracy", "n/a")))
        f1_str  = (f"{m.get('macro_f1','n/a')} ± {m.get('std_macro_f1','n/a')}"
                   if "std_macro_f1" in m else str(m.get("macro_f1", "n/a")))
        logger.info(f"{key:<35} {acc_str:>18} {f1_str:>18}")
    if "delta_B_minus_A" in results:
        d = results["delta_B_minus_A"]
        f1_delta_str = f"{d.get('macro_f1','n/a'):>+18}" if d.get("macro_f1") is not None else "               n/a"
        logger.info(f"{'Δ (B − A)':<35} {d.get('accuracy','n/a'):>+18} {f1_delta_str}")
    logger.info("=" * 75)

    results["meta"] = {
        "fonte_demo":    "real_fapesp" if use_real_demo else "sintetico_correlacionado",
        "n_epochs":      n_epochs,
        "random_seed":   random_seed,
        "seeds":         seeds,
        "demo_features": ["age_norm (birth_year / ref_year=2021)", "sex_binary (M=1, F=0)"],
    }
    return results


def run_pooled_behrt(
    client_loaders: Dict,
    test_loader: DataLoader,
    demographics_by_client: Optional[Dict] = None,
    test_loader_demo: Optional[DataLoader] = None,
    n_epochs: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Pooled baseline: SimplifiedBEHRT treinado no pool de todos os dados de treino.

    ARTEFATO METODOLÓGICO — quantifica o custo de privacidade da arquitetura federada
    isolando-o de diferenças de arquitetura (BEHRT × BEHRT, não BEHRT × RF).
    Nunca deve ser chamado no pipeline de produção nem em run_training.py.

    behrt_pooled_A_sem_demo:    demo_dim=0
    behrt_pooled_B_late_fusion: demo_dim=2
    """
    from mosaicfl.core.evaluation import evaluate

    random_seed   = random_seed if random_seed is not None else RANDOM_SEED
    n_epochs      = n_epochs    if n_epochs    is not None else POOLED_EPOCHS
    use_real_demo = demographics_by_client is not None

    logger.info("=" * 60)
    logger.info("POOLED BASELINE — SimplifiedBEHRT (artefato de pesquisa)")
    logger.info("  Este experimento NUNCA deve rodar em produção.")
    logger.info(f"  Épocas: {n_epochs} (POOLED_EPOCHS) | seed: {random_seed}")
    logger.info(f"  Fonte demográficos: {'dados reais FAPESP' if use_real_demo else 'sintéticos'}")
    logger.info("=" * 60)

    all_seqs, all_lbls, all_demo, all_dia = [], [], [], []
    for cid, (train_loader, _) in client_loaders.items():
        for bx, by, bdia in train_loader:
            all_seqs.append(bx)
            all_lbls.append(by)
            all_dia.append(bdia)

    if use_real_demo:
        for cid, (demo_train_loader, _) in demographics_by_client.items():
            for bx, by, bd, _bdia in demo_train_loader:
                all_demo.append(bd)

    pool_seqs = torch.cat(all_seqs, dim=0)
    pool_lbls = torch.cat(all_lbls, dim=0)
    pool_dia  = torch.cat(all_dia,  dim=0)
    pool_demo = torch.cat(all_demo, dim=0) if all_demo else None
    n_pool    = len(pool_seqs)
    logger.info(f"  Pool total: {n_pool} amostras (BPSP + HSL combinados)")

    results: Dict = {}

    for variant_name, demo_dim in [
        ("behrt_pooled_A_sem_demo",    0),
        ("behrt_pooled_B_late_fusion", 2),
    ]:
        with_demo = (demo_dim > 0)
        logger.info(f"\n[Pooled] Treinando {variant_name} (demo_dim={demo_dim})...")

        torch.manual_seed(random_seed)
        model     = SimplifiedBEHRT(use_cls_token=True, demo_dim=demo_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        counts    = torch.bincount(pool_lbls, minlength=MODEL_CFG.num_classes).float()
        weights   = (n_pool / (MODEL_CFG.num_classes * counts.clamp(min=1))).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        dataset = (
            TensorDataset(pool_seqs, pool_lbls, pool_demo, pool_dia)
            if (with_demo and pool_demo is not None)
            else TensorDataset(pool_seqs, pool_lbls, pool_dia)
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(random_seed),
        )

        model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in loader:
                if with_demo and pool_demo is not None:
                    bx   = batch[0].to(DEVICE)
                    by   = batch[1].to(DEVICE)
                    bd   = batch[2].to(DEVICE)
                    bdia = batch[3].to(DEVICE)
                    logits = model(bx, demographics=bd, dia_relativo=bdia)
                else:
                    bx   = batch[0].to(DEVICE)
                    by   = batch[1].to(DEVICE)
                    bdia = batch[2].to(DEVICE)
                    logits = model(bx, dia_relativo=bdia)
                loss = criterion(logits, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(f"    época {epoch+1}/{n_epochs} loss={epoch_loss/len(loader):.4f}")

        if with_demo and test_loader_demo is not None:
            model.eval()
            all_preds, all_labels_eval = [], []
            eval_loss, eval_n = 0.0, 0
            with torch.no_grad():
                for batch in test_loader_demo:
                    bx   = batch[0].to(DEVICE)
                    by   = batch[1].to(DEVICE)
                    bd   = batch[2].to(DEVICE) if len(batch) >= 3 else None
                    bdia = batch[3].to(DEVICE) if len(batch) >= 4 else None
                    logits = model(bx, demographics=bd, dia_relativo=bdia)
                    eval_loss += nn.CrossEntropyLoss()(logits, by).item() * by.size(0)
                    eval_n += by.size(0)
                    all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
                    all_labels_eval.extend(by.cpu().tolist())
            from sklearn.metrics import accuracy_score, f1_score
            metrics: Dict = {
                "accuracy": round(float(accuracy_score(all_labels_eval, all_preds)), 4),
                "macro_f1": round(float(f1_score(all_labels_eval, all_preds,
                                                  average="macro", zero_division=0)), 4),
                "loss":     round(eval_loss / eval_n, 4) if eval_n > 0 else None,
            }
        else:
            try:
                report = evaluate(model, test_loader, class_labels=MODEL_CFG.class_labels,
                                  device=str(DEVICE), temperature=1.0)
                metrics = {
                    "accuracy":     round(report.accuracy, 4),
                    "macro_f1":     round(report.macro_f1, 4),
                    "macro_auc":    round(report.macro_auc, 4) if report.macro_auc else None,
                    "ece":          round(report.calibration.ece, 4),
                    "per_class_f1": {
                        lbl: round(m.f1, 4)
                        for lbl, m in report.per_class.items()
                    },
                }
            except Exception as exc:
                logger.warning(f"evaluate() falhou para {variant_name}: {exc}")
                metrics = {"erro": str(exc)}

        metrics["demo_dim"]  = demo_dim
        metrics["n_epochs"]  = n_epochs
        metrics["n_pool"]    = n_pool
        metrics["descricao"] = (
            "BEHRT pooled baseline — artefato de pesquisa. "
            "Dados BPSP+HSL combinados. Nunca implantado."
        )
        results[variant_name] = metrics
        logger.info(f"  {variant_name}: Acc={metrics.get('accuracy','n/a')} | F1={metrics.get('macro_f1','n/a')}")

    a = results.get("behrt_pooled_A_sem_demo",    {})
    b = results.get("behrt_pooled_B_late_fusion",  {})
    if "accuracy" in a and "accuracy" in b:
        results["delta_B_minus_A"] = {
            "accuracy": round(b["accuracy"] - a["accuracy"], 4),
            "macro_f1": round(b.get("macro_f1", 0) - a.get("macro_f1", 0), 4),
        }

    logger.info("\n" + "=" * 70)
    logger.info("RESULTADO POOLED BASELINE")
    logger.info("=" * 70)
    logger.info(f"{'Variante':<40} {'Accuracy':>8} {'F1 Macro':>8}")
    logger.info("-" * 60)
    for key in ["behrt_pooled_A_sem_demo", "behrt_pooled_B_late_fusion"]:
        m = results.get(key, {})
        logger.info(f"{key:<40} {m.get('accuracy','n/a'):>8} {m.get('macro_f1','n/a'):>8}")
    if "delta_B_minus_A" in results:
        d = results["delta_B_minus_A"]
        logger.info(f"{'Δ (B − A)':<40} {d.get('accuracy','n/a'):>+8} {d.get('macro_f1','n/a'):>+8}")
    logger.info("=" * 70)

    results["meta"] = {
        "tipo":        "behrt_pooled_baseline",
        "aviso":       "Artefato metodológico. Nunca executar em produção.",
        "n_epochs":    n_epochs,
        "random_seed": random_seed,
        "n_pool":      n_pool,
        "fonte_demo":  "real_fapesp" if use_real_demo else "sintetico",
    }
    return results
