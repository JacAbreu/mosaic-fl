"""
federated.py — Utilitários de agregação federada.

Funções de agregação ponderada usadas por ambos os adapters
(produção e experimento) para consolidar métricas dos clientes.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from collections import OrderedDict


def _weighted_average(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> Dict[str, float]:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {key: 0.0}
    return {key: sum(n * m.get(key, 0.0) for n, m in metrics) / total}


def weighted_average_accuracy(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """Média ponderada de accuracy — para evaluate_metrics_aggregation_fn."""
    return _weighted_average(metrics, "accuracy")


def weighted_average_evaluate_metrics(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Agrega accuracy, f1_macro e per_class_f1 — para evaluate_metrics_aggregation_fn
    no Caminho B (produção). F1 é calculado localmente em cada cliente (nunca expõe
    predições/labels brutos ao servidor) e chega serializado em "per_class_f1_json"
    (flwr.common.Metrics só aceita valores escalares — dict/list bruto quebraria a
    agregação). Reserializa o resultado agregado do mesmo jeito, pro chamador
    (aggregate_evaluate em core.py) desserializar de volta.

    labels=range(num_classes) já garantido no cliente (client.py) — todos os
    per_class_f1 chegam com o mesmo tamanho, mesmo com hospitais com distribuições
    de classe muito diferentes (BPSP x HSL).
    """
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    result: Dict[str, Any] = {
        "accuracy": sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total,
        "f1_macro": sum(n * m.get("f1_macro", 0.0) for n, m in metrics) / total,
    }

    per_class_lists = [
        (n, json.loads(m["per_class_f1_json"]))
        for n, m in metrics if "per_class_f1_json" in m
    ]
    if per_class_lists:
        num_classes = len(per_class_lists[0][1])
        per_class_f1 = [
            sum(n * pc[i] for n, pc in per_class_lists) / total
            for i in range(num_classes)
        ]
        result["per_class_f1_json"] = json.dumps(per_class_f1)

    # Padrões pro RAG (rag_patterns_json) — só presentes na rodada em que o servidor
    # pediu (config extract_rag_patterns). Diferente de accuracy/F1, NÃO se faz média —
    # são perfis independentes por hospital, concatenados numa base de conhecimento
    # combinada (cada hospital contribui os seus próprios perfis prototípicos).
    all_patterns: list = []
    for _, m in metrics:
        if "rag_patterns_json" in m:
            all_patterns.extend(json.loads(m["rag_patterns_json"]))
    if all_patterns:
        result["rag_patterns_json"] = json.dumps(all_patterns)

    calibration = aggregate_calibration(metrics)
    if calibration:
        result.update(calibration)

    return result


def aggregate_calibration(metrics: List[Tuple[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Agrega calibradores locais (ajustados em FedProxClient._fit_local_calibrator,
    só presentes na rodada em que o servidor pediu via config `calibrate`) num único
    calibrador federado. Retorna None se nenhum cliente enviou dado de calibração
    (rodadas normais, sem o pedido do servidor).

    temperature: média ponderada do escalar T entre clientes (mesmo princípio do
    FedTemp — Maddock, Cormode & Maple, "Private Federated Multiclass Post-hoc
    Calibration", preprint arXiv:2510.01987, 2025 ***— sem revisão por pares, ver
    docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md §9.1).

    isotonic: concatena os breakpoints pós-PAV (X_thresholds/y_thresholds — estatísticas
    comprimidas, não dado bruto por amostra) de cada cliente por classe, e refaz o fit
    sobre o conjunto agregado — mesmo espírito do histograma agregável do Cormode &
    Markov ("Federated Calibration and Evaluation of Binary Classifiers", VLDB 2023,
    §9.2), adaptado para thresholds isotônicos em vez de contagens de histograma.
    """
    calib_entries = [(n, m) for n, m in metrics if "calibration_method" in m]
    if not calib_entries:
        return None

    method = calib_entries[0][1]["calibration_method"]

    if method == "temperature":
        total = sum(n for n, m in calib_entries if "temperature" in m)
        if total == 0:
            return None
        temperature = sum(n * m["temperature"] for n, m in calib_entries if "temperature" in m) / total
        return {
            "calibration_method": "temperature",
            "temperature": temperature,
        }

    if method == "isotonic":
        per_class_pooled: Dict[int, Tuple[list, list]] = {}
        num_classes = 0
        for _, m in calib_entries:
            if "isotonic_thresholds_json" not in m:
                continue
            thresholds = json.loads(m["isotonic_thresholds_json"])
            num_classes = max(num_classes, len(thresholds))
            for c, (x_list, y_list) in enumerate(thresholds):
                x_all, y_all = per_class_pooled.setdefault(c, ([], []))
                x_all.extend(x_list)
                y_all.extend(y_list)
        if not per_class_pooled:
            return None
        pooled_per_class = [list(per_class_pooled.get(c, ([], []))) for c in range(num_classes)]
        return {
            "calibration_method": "isotonic",
            "isotonic_pooled_thresholds_json": json.dumps(pooled_per_class),
            "isotonic_num_classes": num_classes,
        }

    return None


def weighted_average_loss(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """Média ponderada de loss — para fit_metrics_aggregation_fn."""
    return _weighted_average(metrics, "loss")


# Alias mantido para compatibilidade com código existente
weighted_average = weighted_average_accuracy


def get_evaluate_fn(test_loader: "torch.utils.data.DataLoader[Any]") -> Callable:
    """
    Retorna a função de avaliação global usada pelo servidor a cada rodada.

    Compatível com evaluate_fn do Flower — avalia o modelo global no
    test_loader fornecido e retorna (loss, {"accuracy": float}).
    """
    from mosaicfl.core.model import SimplifiedBEHRT
    from mosaicfl.core.config import RUNTIME_CFG

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict,
    ) -> Tuple[float, Dict]:
        model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for batch_x, batch_y, *rest in test_loader:
                batch_x = batch_x.to(RUNTIME_CFG.device)
                batch_y = batch_y.to(RUNTIME_CFG.device)
                batch_dia = rest[0].to(RUNTIME_CFG.device) if rest else None
                logits = model(batch_x, dia_relativo=batch_dia)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                _, predicted = torch.max(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"  [Servidor] Rodada {server_round:>3} | "
            f"Loss global: {avg_loss:.4f} | Acurácia global: {accuracy:.4f}"
        )
        return avg_loss, {"accuracy": accuracy, "rodada": server_round}

    return evaluate
