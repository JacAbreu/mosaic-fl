"""
Servidor Flower com estratégia FedProx e critério de parada antecipada — VERSÃO CORRIGIDA.

Mudanças principais:
  1. CustomFedProxStrategy herda de FedProx e conecta ConvergenceTracker,
     permitindo parada antecipada quando a acurácia global estabiliza.
  2. Histórico de treinamento é populado a cada rodada (rounds, accuracy, loss).
  3. Persistência automática do modelo global ao final (ou a cada rodada).
  4. Callback opcional on_converged para ações pós-convergência (ex: salvar RAG).
"""
import flwr as fl
from flwr.server.strategy import FedProx
from typing import List, Tuple, Dict, Optional, Callable
from collections import OrderedDict
import numpy as np
import torch
import json
import os
from datetime import datetime

from .model_v2 import SimplifiedBEHRT
from .config import (
    CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD, DEVICE,
    FRACTION_EVALUATE, FRACTION_FIT,
    MIN_AVAILABLE_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_FIT_CLIENTS,
    NUM_CLIENTS, NUM_ROUNDS, PROXIMAL_MU,
)

# Tamanho real do state_dict (upload + download por cliente = × 2)
_MODEL_SIZE_MB: float = round(
    sum(v.numel() * v.element_size() for v in SimplifiedBEHRT(use_cls_token=True).state_dict().values())
    / (1024 ** 2),
    3,
)


class ConvergenceTracker:
    def __init__(self, threshold: float = CONVERGENCE_THRESHOLD, patience: int = CONVERGENCE_PATIENCE):
        self.threshold = threshold
        self.patience = patience
        self.history = []
        self.stable_count = 0
        self.converged_round = None

    def check(self, accuracy: float) -> bool:
        self.history.append(accuracy)
        if len(self.history) < 2:
            return False
        delta = abs(self.history[-1] - self.history[-2])
        if delta < self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        if self.stable_count >= self.patience and self.converged_round is None:
            self.converged_round = len(self.history)
        return self.stable_count >= self.patience

    def reset(self) -> None:
        self.history.clear()
        self.stable_count = 0
        self.converged_round = None


def _weighted_average(metrics: List[Tuple[int, Dict]], key: str) -> Dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {key: 0.0}
    return {key: sum(n * m.get(key, 0.0) for n, m in metrics) / total}


def weighted_average_accuracy(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Média ponderada de accuracy — para evaluate_metrics_aggregation_fn."""
    return _weighted_average(metrics, "accuracy")


def weighted_average_loss(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Média ponderada de loss — para fit_metrics_aggregation_fn."""
    return _weighted_average(metrics, "loss")


# Alias para compatibilidade com código existente
weighted_average = weighted_average_accuracy


class CustomFedProxStrategy(FedProx):
    """
    Estratégia FedProx customizada com:
      - ConvergenceTracker integrado (parada antecipada)
      - Histórico de treinamento populado
      - Persistência de modelo global
    """
    def __init__(self, tracker: ConvergenceTracker, history: Dict,
                 save_dir: str = "checkpoints", on_converged: Optional[Callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.history = history
        self.save_dir = save_dir
        self.on_converged = on_converged
        os.makedirs(save_dir, exist_ok=True)
        self._round_counter = 0

    def aggregate_fit(self, server_round: int, results, failures):
        """Salva pesos agregados em disco imediatamente após cada rodada."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = SimplifiedBEHRT(use_cls_token=True)
            keys = list(model.state_dict().keys())
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})
            model.load_state_dict(state_dict, strict=False)
            path = os.path.join(self.save_dir, f"round_{server_round}.pt")
            torch.save(model.state_dict(), path)
            self.history["last_checkpoint"] = path
            print(f"   💾 Checkpoint salvo: {path}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Chamado após avaliação dos clientes — usado para rastrear convergência."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None and len(aggregated) == 2:
            loss, metrics = aggregated
            accuracy = metrics.get("accuracy", 0.0)
            self._round_counter = server_round

            self.history["rounds"].append(server_round)
            self.history["accuracy"].append(accuracy)
            self.history["communication_mb"].append(round(len(results) * _MODEL_SIZE_MB * 2, 3))

            if self.tracker.check(accuracy):
                print(f"\n🎯 CONVERGÊNCIA ATINGIDA na rodada {server_round}!")
                print(f"   Acurácia: {accuracy:.4f} | Δ < {self.tracker.threshold} por {self.tracker.patience} rodadas.")
                if self.on_converged:
                    self.on_converged(server_round, accuracy, self.history)
                self._save_checkpoint(server_round, final=True)
                raise StopIteration(f"Convergência atingida na rodada {server_round}")

        return aggregated

    def _save_checkpoint(self, server_round: int, final: bool = False) -> None:
        """Cria cópia nomeada 'final.pt' do último checkpoint ao convergir."""
        if not final:
            return
        last = self.history.get("last_checkpoint")
        if last and os.path.exists(last):
            import shutil
            final_path = os.path.join(self.save_dir, "final.pt")
            shutil.copy2(last, final_path)
            self.history["last_checkpoint"] = final_path
            print(f"   💾 Checkpoint final: {final_path}")


def get_evaluate_fn(test_loader) -> Callable:
    """
    Retorna a função de avaliação global usada pelo servidor a cada rodada.
    """
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict,
    ) -> Tuple[float, Dict]:
        model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                _, predicted = torch.max(logits, dim=1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print(
            f"  [Servidor] Rodada {server_round:>3} | "
            f"Loss global: {avg_loss:.4f} | Acurácia global: {accuracy:.4f}"
        )
        return avg_loss, {"accuracy": accuracy, "rodada": server_round}

    return evaluate


def start_server(
    num_rounds: int = NUM_ROUNDS,
    num_clients: int = NUM_CLIENTS,
    test_loader=None,
    on_converged: Optional[Callable] = None,
) -> Tuple["CustomFedProxStrategy", ConvergenceTracker, Dict]:
    """
    Monta a estratégia FedProx com convergência integrada.

    Se test_loader for fornecido, a avaliação global real é ativada.
    """
    evaluate_fn = get_evaluate_fn(test_loader) if test_loader is not None else None
    tracker = ConvergenceTracker()
    history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}

    strategy = CustomFedProxStrategy(
        tracker=tracker,
        history=history,
        on_converged=on_converged,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        proximal_mu=PROXIMAL_MU,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
        fit_metrics_aggregation_fn=weighted_average_loss,
    )

    print(f"Iniciando servidor FedProx por até {num_rounds} rodadas...")
    print(f"Mu proximal: {PROXIMAL_MU} | Clientes mínimos: {MIN_FIT_CLIENTS}")
    print(f"Convergência: Δ < {CONVERGENCE_THRESHOLD} por {CONVERGENCE_PATIENCE} rodadas.")
    if evaluate_fn:
        print("Avaliação global: ATIVA (test_loader fornecido)")
    else:
        print("Avaliação global: INATIVA")

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
        )
    except StopIteration as e:
        print(f"\nTreinamento interrompido por convergência antecipada: {e}")

    return strategy, tracker, history


if __name__ == "__main__":
    strategy, tracker, history = start_server()