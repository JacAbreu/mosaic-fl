"""
Servidor Flower com estratégia FedProx e criterio de parada antecipada (CustomFedProxStrategy).

Agrega pesos dos clientes via FedAvg ponderado, rastreia convergencia da acuracia
global e para automaticamente quando delta < threshold por patience rodadas consecutivas.
Salva checkpoint a cada round e checkpoint final ao convergir.
"""
import logging
import flwr as fl

logger = logging.getLogger(__name__)
from flwr.server.strategy import FedProx
from typing import List, Tuple, Dict, Optional, Callable
from collections import OrderedDict
import numpy as np
import torch
import json
import os
from datetime import datetime

from .model_v2 import SimplifiedBEHRT
from .config import FED_CFG, RUNTIME_CFG

# Tamanho real do state_dict (upload + download por cliente = × 2)
_MODEL_SIZE_MB: float = round(
    sum(v.numel() * v.element_size() for v in SimplifiedBEHRT(use_cls_token=True).state_dict().values())
    / (1024 ** 2),
    3,
)


class ConvergenceTracker:
    def __init__(self, threshold: float = FED_CFG.convergence_threshold, patience: int = FED_CFG.convergence_patience):
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
            logger.info("checkpoint_saved", extra={"round": server_round, "path": path})

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
                logger.info(
                    "convergence_reached",
                    extra={"round": server_round, "accuracy": accuracy,
                           "threshold": self.tracker.threshold, "patience": self.tracker.patience},
                )
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
            logger.info("checkpoint_final_saved", extra={"path": final_path})


def get_evaluate_fn(test_loader) -> Callable:
    """
    Retorna a função de avaliação global usada pelo servidor a cada rodada.
    """
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
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(RUNTIME_CFG.device), batch_y.to(RUNTIME_CFG.device)
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
    num_rounds: int = FED_CFG.num_rounds,
    num_clients: int = FED_CFG.num_clients,
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
        fraction_fit=FED_CFG.fraction_fit,
        fraction_evaluate=FED_CFG.fraction_evaluate,
        min_fit_clients=FED_CFG.min_fit_clients,
        min_evaluate_clients=FED_CFG.min_evaluate_clients,
        min_available_clients=FED_CFG.min_available_clients,
        proximal_mu=FED_CFG.proximal_mu,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
        fit_metrics_aggregation_fn=weighted_average_loss,
    )

    print(f"Iniciando servidor FedProx por até {num_rounds} rodadas...")
    print(f"Mu proximal: {FED_CFG.proximal_mu} | Clientes mínimos: {FED_CFG.min_fit_clients}")
    print(f"Convergência: Δ < {FED_CFG.convergence_threshold} por {FED_CFG.convergence_patience} rodadas.")
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