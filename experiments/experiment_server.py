"""
experiment_server.py — Adapter de servidor FL para experimentos locais.

Não é destinado a deploy. Usa mosaicfl.core como domínio e Flower gRPC
localmente para simular rounds federados durante a pesquisa do TCC.

Diferenças em relação ao adapter de produção (infrastructure/mosaicfl_server/):
  - Sem TLS, sem health server, sem ChromaDB
  - History populado para análise pós-treino
  - Para treinamento via StopIteration ao convergir
"""
import logging
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.server.strategy import FedProx

import torch

from mosaicfl.core.model_v2 import SimplifiedBEHRT
from mosaicfl.core.config import FED_CFG, RUNTIME_CFG
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.federated import weighted_average_accuracy, weighted_average_loss

logger = logging.getLogger(__name__)

_MODEL_SIZE_MB: float = round(
    sum(v.numel() * v.element_size() for v in SimplifiedBEHRT(use_cls_token=True).state_dict().values())
    / (1024 ** 2),
    3,
)


class CustomFedProxStrategy(FedProx):
    """
    Estratégia FedProx customizada para experimentos:
      - ConvergenceTracker integrado (parada antecipada)
      - Histórico de treinamento populado para análise
      - Persistência de modelo global
    """

    def __init__(
        self,
        tracker: ConvergenceTracker,
        history: Dict,
        save_dir: str = "checkpoints",
        on_converged: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
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
                    extra={
                        "round": server_round,
                        "accuracy": accuracy,
                        "threshold": self.tracker.threshold,
                        "patience": self.tracker.patience,
                    },
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


def start_server(
    num_rounds: int = FED_CFG.num_rounds,
    num_clients: int = FED_CFG.num_clients,
    test_loader=None,
    on_converged: Optional[Callable] = None,
) -> Tuple["CustomFedProxStrategy", ConvergenceTracker, Dict]:
    """
    Monta a estratégia FedProx com convergência integrada e inicia servidor local.

    Se test_loader for fornecido, a avaliação global real é ativada.
    """
    from mosaicfl.core.federated import get_evaluate_fn
    evaluate_fn = get_evaluate_fn(test_loader) if test_loader is not None else None
    tracker = ConvergenceTracker(
        threshold=FED_CFG.convergence_threshold,
        patience=FED_CFG.convergence_patience,
    )
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
