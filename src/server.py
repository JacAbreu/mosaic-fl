"""
Servidor Flower com estratégia FedProx e critério de parada antecipada.

Correção (item 7):
  - get_evaluate_fn agora carrega os parâmetros agregados no modelo global,
    roda um loop de avaliação real no test_loader e retorna loss + métricas.
  - start_server recebe test_loader opcional e passa evaluate_fn para FedProx,
    conectando a avaliação global ao ciclo de treinamento federado.
"""
import flwr as fl
from flwr.server.strategy import FedProx
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import numpy as np
import torch
import json

from model import SimplifiedBEHRT
from config import *


class ConvergenceTracker:
    def __init__(self, threshold: float = CONVERGENCE_THRESHOLD, patience: int = CONVERGENCE_PATIENCE):
        self.threshold = threshold
        self.patience = patience
        self.history = []
        self.stable_count = 0

    def check(self, accuracy: float) -> bool:
        self.history.append(accuracy)
        if len(self.history) < 2:
            return False
        delta = abs(self.history[-1] - self.history[-2])
        if delta < self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        return self.stable_count >= self.patience


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Agrega métricas ponderadas pelo número de amostras."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(test_loader):
    """
    Retorna a função de avaliação global usada pelo servidor a cada rodada.

    O Flower chama evaluate(server_round, parameters, config) após agregar
    os pesos dos clientes. Aqui carregamos esses parâmetros em um modelo
    local (nunca compartilhado com clientes) e avaliamos no conjunto de
    teste global, que só o servidor conhece — mantendo a privacidade do FL.

    Args:
        test_loader: DataLoader com o conjunto de teste global (holdout).

    Returns:
        Callable com assinatura esperada pelo Flower:
            (int, NDArrays, Dict) -> Tuple[float, Dict[str, Scalar]]
    """
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict,
    ) -> Tuple[float, Dict]:

        # 1. Reconstrói o modelo global com os pesos agregados desta rodada
        model = SimplifiedBEHRT().to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # 2. Avaliação no conjunto de teste global
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item()

                _, predicted = torch.max(logits, dim=1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = loss_sum / len(test_loader)
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
    test_loader=None,          # DataLoader do conjunto de teste global (opcional)
):
    """
    Monta a estratégia FedProx e retorna (strategy, tracker, history).

    Se test_loader for fornecido, a avaliação global real é ativada —
    o servidor avaliará o modelo agregado após cada rodada.
    Sem test_loader, a avaliação global fica desligada (útil em testes rápidos).
    """
    # Avaliação global: só ativa se um test_loader for passado
    evaluate_fn = get_evaluate_fn(test_loader) if test_loader is not None else None

    strategy = FedProx(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        proximal_mu=PROXIMAL_MU,
        evaluate_fn=evaluate_fn,                          # ← conectado aqui
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    tracker = ConvergenceTracker()

    # Histórico para análise do Experimento 5
    history = {"rounds": [], "accuracy": [], "communication_mb": []}

    print(f"Iniciando servidor FedProx por {num_rounds} rodadas...")
    print(f"Mu proximal: {PROXIMAL_MU} | Clientes mínimos: {MIN_FIT_CLIENTS}")
    if evaluate_fn:
        print("Avaliação global: ATIVA (test_loader fornecido)")
    else:
        print("Avaliação global: INATIVA (passe test_loader para ativar)")

    return strategy, tracker, history


if __name__ == "__main__":
    strategy, tracker, history = start_server()
    # Para rodar de verdade com avaliação global:
    # strategy, tracker, history = start_server(test_loader=meu_test_loader)
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     strategy=strategy,
    #     config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    # )
