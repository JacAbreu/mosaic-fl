"""fit_config_mixin.py — Configuração de round (leitura de config dinâmica) e carregamento de pesos agregados."""
import logging
import time
from collections import OrderedDict
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


class _FitConfigMixin:
    """Requer os atributos definidos em ProductionFedProxStrategy.__init__ (config_loader, global_model,
    on_round_start, proximal_mu) e o método _start_round_watchdog (de _WatchdogMixin)."""

    def configure_fit(
        self, server_round: int, parameters, client_manager
    ) -> List[Tuple]:
        """
        Chamado pelo Flower antes de cada round de treino.

        Lê config dinâmica do PostgreSQL (ou fallback arquivo) e aplica antes
        de delegar a seleção de clientes ao FedProx padrão.
        """
        runtime = self.config_loader.load(server_round)

        if runtime.get("stop", False):
            logger.info("round_stopped", extra={"round": server_round, "reason": "config_stop"})
            self.should_stop = True
            return []

        if "proximal_mu" in runtime and runtime["proximal_mu"] is not None:
            new_mu = float(runtime["proximal_mu"])
            if new_mu != self.proximal_mu:
                logger.info(
                    "proximal_mu_updated",
                    extra={"round": server_round, "old_mu": self.proximal_mu, "new_mu": new_mu},
                )
                self.proximal_mu = new_mu

        pause = float(runtime.get("pause_seconds", 0) or 0)
        if pause > 0:
            logger.info("round_paused", extra={"round": server_round, "pause_seconds": pause})
            time.sleep(pause)

        if self.on_round_start is not None:
            try:
                self.on_round_start(server_round, runtime)
            except Exception as e:
                logger.warning("round_start_callback_error", extra={"round": server_round, "error": str(e)})

        self._start_round_watchdog(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def _load_global_weights(self, parameters) -> None:
        """Carrega pesos agregados no modelo global (compatível com client)."""
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
