"""
strategy.py
Estratégia FedProx de produção (mosaicfl.v2) com checkpoint e convergência.
"""
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import flwr as fl
import torch

from mosaicfl.v2.config import CONVERGENCE_THRESHOLD, CONVERGENCE_PATIENCE
from mosaicfl.v2.server_v2 import weighted_average

CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))

logger = logging.getLogger(__name__)


class ConvergenceTracker:
    """Rastreia convergência da acurácia global."""

    def __init__(
        self,
        threshold: float = CONVERGENCE_THRESHOLD,
        patience: int = CONVERGENCE_PATIENCE,
    ):
        self.threshold = threshold
        self.patience = patience
        self.history: List[float] = []
        self.converged_round: Optional[int] = None

    def check(self, accuracy: float) -> bool:
        self.history.append(accuracy)
        if len(self.history) < self.patience + 1:
            return False
        recent = self.history[-(self.patience + 1) :]
        deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        converged = all(d < self.threshold for d in deltas)
        if converged and self.converged_round is None:
            self.converged_round = len(self.history)
            logger.info("Convergência atingida na rodada %s", self.converged_round)
        return converged

    def reset(self):
        self.history.clear()
        self.converged_round = None


class ProductionFedProxStrategy(fl.server.strategy.FedProx):
    """
    FedProx adaptado para produção:
      - Checkpoint do modelo global a cada rodada
      - Exporta métricas para JSON (consumidas pelo scheduler)
      - Rastreia convergência
    """

    def __init__(self, global_model: torch.nn.Module, *args, **kwargs):
        kwargs.setdefault("evaluate_metrics_aggregation_fn", weighted_average)
        kwargs.setdefault("fit_metrics_aggregation_fn", weighted_average)
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.tracker = ConvergenceTracker()
        self.round_counter = 0
        self.should_stop = False
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_global_weights(self, parameters) -> None:
        """Carrega pesos agregados no modelo global (compatível com client_v2)."""
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.global_model.state_dict().keys(), parameters)
            }
        )
        missing, unexpected = self.global_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Checkpoint: chaves não carregadas: %s", missing)
        if unexpected:
            logger.debug("Checkpoint: chaves inesperadas: %s", unexpected)

    def aggregate_fit(self, server_round, results, failures):
        """Agrega pesos e salva checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._load_global_weights(aggregated_parameters)
            checkpoint_path = CHECKPOINT_DIR / f"round_{server_round}.pt"
            torch.save(self.global_model.state_dict(), checkpoint_path)
            logger.info("Checkpoint salvo: %s", checkpoint_path)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas e detecta convergência."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracy = aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0
        self.tracker.check(accuracy)
        self.round_counter = server_round

        metrics_file = LOG_DIR / f"round_{server_round}_metrics.json"
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round": server_round,
                        "loss": aggregated_loss,
                        "accuracy": accuracy,
                        "timestamp": datetime.now().isoformat(),
                        "converged": self.tracker.converged_round is not None,
                        "convergence_round": self.tracker.converged_round,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning("Erro ao salvar métricas: %s", e)

        if self.tracker.converged_round is not None:
            self.should_stop = True
            logger.info("Convergência detectada. Servidor encerrará após esta rodada.")

        return aggregated_loss, aggregated_metrics
