"""
experiment_server.py — Adapter de servidor FL para experimentos locais.

Não é destinado a deploy. Usa mosaicfl.core como domínio e Flower gRPC
localmente para simular rounds federados durante a pesquisa do TCC.

Diferenças em relação ao adapter de produção (infrastructure/mosaicfl_server/):
  - Sem TLS, sem health server, sem ChromaDB
  - History populado para análise pós-treino
  - Para treinamento via StopIteration ao convergir
"""
import functools
import logging
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.server.strategy import FedProx

import torch

from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.config import FED_CFG, RUNTIME_CFG
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.federated import weighted_average_accuracy, weighted_average_loss
from infrastructure.shared.checkpoint_store import CheckpointStore, get_checkpoint_store

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=None)
def _model_size_mb() -> float:
    """Tamanho do modelo em MB — calculado uma vez na primeira chamada."""
    return round(
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
        checkpoint_store: CheckpointStore,
        vocab: Optional[Dict[str, int]] = None,
        on_converged: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.history = history
        self.checkpoint_store = checkpoint_store
        self.vocab = vocab or {}
        self.on_converged = on_converged
        self._round_counter = 0

    def aggregate_fit(self, server_round: int, results, failures):
        """Agrega pesos e persiste checkpoint no store após cada rodada."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = SimplifiedBEHRT(use_cls_token=True)
            keys = list(model.state_dict().keys())
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})
            model.load_state_dict(state_dict, strict=False)

            last_acc = self.history["accuracy"][-1] if self.history["accuracy"] else 0.0
            self.checkpoint_store.save(
                round_num=server_round,
                state_dict=state_dict,
                vocab=self.vocab,
                accuracy=last_acc,
            )
            self.history["last_checkpoint"] = f"db:round_{server_round}"
            logger.info("checkpoint_saved", extra={"round": server_round, "store": type(self.checkpoint_store).__name__})

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
            self.history["communication_mb"].append(round(len(results) * _model_size_mb() * 2, 3))

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
        """No-op: checkpoints são persistidos a cada rodada em aggregate_fit."""
        if final:
            logger.info(
                "checkpoint_final_noted",
                extra={"round": server_round, "store": type(self.checkpoint_store).__name__},
            )


def start_server(
    num_rounds: int = FED_CFG.num_rounds,
    num_clients: int = FED_CFG.num_clients,
    test_loader=None,
    vocab: Optional[Dict[str, int]] = None,
    on_converged: Optional[Callable] = None,
) -> Tuple["CustomFedProxStrategy", ConvergenceTracker, Dict]:
    """
    Monta a estratégia FedProx com convergência integrada e inicia servidor local.

    Se test_loader for fornecido, a avaliação global real é ativada.
    Checkpoints são persistidos via get_checkpoint_store() (SQLite em experimentos,
    PostgreSQL quando FL_DB_URL estiver configurado).
    """
    from mosaicfl.core.federated import get_evaluate_fn
    evaluate_fn = get_evaluate_fn(test_loader) if test_loader is not None else None
    tracker = ConvergenceTracker(
        threshold=FED_CFG.convergence_threshold,
        patience=FED_CFG.convergence_patience,
    )
    history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
    checkpoint_store = get_checkpoint_store(RUNTIME_CFG.db_url)

    strategy = CustomFedProxStrategy(
        tracker=tracker,
        history=history,
        checkpoint_store=checkpoint_store,
        vocab=vocab or {},
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
