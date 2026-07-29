"""fit_config_mixin.py — Configuração de round (leitura de config dinâmica) e carregamento de pesos agregados."""
import json
import logging
import time
from collections import OrderedDict
from typing import List, Tuple

import torch
from flwr.common import parameters_to_ndarrays

logger = logging.getLogger(__name__)


class _FitConfigMixin:
    """Requer os atributos definidos em ProductionFedProxStrategy.__init__ (config_loader, global_model,
    on_round_start, proximal_mu) e o método _start_round_watchdog (de _WatchdogMixin)."""

    def _inject_current_vocab(self, instructions: List[Tuple]) -> List[Tuple]:
        """Sobrescreve vocab_json em cada (ClientProxy, Ins) com self.vocab — fonte
        única e mutável (atualizada só em initialize_parameters(), antes da Rodada 1,
        ver strategy/core.py::_discover_and_curate_vocab). Os lambdas
        on_fit_config_fn/on_evaluate_config_fn (superlink.py) não têm como referenciar
        self.vocab diretamente (a estratégia ainda não existe no momento em que são
        definidos) — por isso a injeção acontece aqui, depois que o Flower já monta
        a config base a partir deles."""
        for _, ins in instructions:
            ins.config["vocab_json"] = json.dumps(self.vocab)
        return instructions

    def _inject_rag_patterns(self, instructions: List[Tuple]) -> List[Tuple]:
        """Reenvia os padrões do RAG (self._last_rag_patterns_json, cacheados em
        _build_rag_knowledge_base) pra cada cliente avaliar Precision@k localmente
        (client.py::evaluate(), ver mosaicfl.core.rag.precision.eval_precision_at_k).
        Ausente/None antes da 1ª vez que algum cliente enviou padrões — não injeta
        nada, cliente não recebe a chave e pula o cálculo (Precision@k só fica
        disponível a partir da 2ª rodada de avaliação em diante)."""
        if not self._last_rag_patterns_json:
            return instructions
        for _, ins in instructions:
            ins.config["rag_patterns_json"] = self._last_rag_patterns_json
        return instructions

    def _inject_class_weight_overrides(self, instructions: List[Tuple], runtime: dict) -> List[Tuple]:
        """Peso de classe explícito (cost-sensitive learning, Strategy pattern em
        mosaicfl.core.class_weighting — ver docs/pesquisa_baseline_implementacao_fontes_
        bibliograficas.md, seção 14). Lido de clinical.fl_orchestration_config (mesma
        fonte de proximal_mu/pause_seconds/stop) e empurrado idêntico pros dois
        hospitais — sem isso, cada hospital precisaria de .env sincronizado
        manualmente, e os dois bancos são locais/separados por desenho (dado clínico
        nunca sai do hospital). Só o valor da PRIMEIRA rodada tem efeito de fato
        (FedProxClient._ensure_data só carrega/computa uma vez), mas injeta a cada
        rodada por simplicidade e paridade com _inject_current_vocab. Ausente/None =
        nenhum override — cliente cai no fallback de FED_CFG.class_weight_overrides
        (env, dev/simulação)."""
        overrides_json = runtime.get("class_weight_overrides_json")
        if overrides_json is None:
            return instructions
        for _, ins in instructions:
            ins.config["class_weight_overrides_json"] = overrides_json
        return instructions

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
        instructions = super().configure_fit(server_round, parameters, client_manager)
        instructions = self._inject_current_vocab(instructions)
        return self._inject_class_weight_overrides(instructions, runtime)

    def configure_evaluate(
        self, server_round: int, parameters, client_manager
    ) -> List[Tuple]:
        """Mesma injeção de self.vocab que configure_fit() já faz — sem isso, a
        Rodada de avaliação usaria o vocab capturado por closure em superlink.py
        (achado 2026-07-26, ver initialize_parameters/_discover_and_curate_vocab).
        Também injeta class_weight_overrides_json (leitura própria de
        clinical.fl_orchestration_config — configure_fit já rodou nesta mesma rodada,
        mas configure_evaluate pode ser a PRIMEIRA chamada a chegar no cliente
        dependendo da ordem do protocolo, e _ensure_data só carrega uma vez)."""
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        instructions = self._inject_current_vocab(instructions)
        instructions = self._inject_rag_patterns(instructions)
        runtime = self.config_loader.load(server_round)
        return self._inject_class_weight_overrides(instructions, runtime)

    def _load_global_weights(self, parameters) -> None:
        """Carrega pesos agregados no modelo global (compatível com client).

        parameters chega como flwr.common.Parameters (retorno de
        super().aggregate_fit()), não como lista de ndarrays — precisa
        converter antes de iterar, senão `zip()` falha com
        "'Parameters' object is not iterable".
        """
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.global_model.state_dict().keys(), ndarrays)
            }
        )
        missing, unexpected = self.global_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Checkpoint: chaves não carregadas: %s", missing)
        if unexpected:
            logger.debug("Checkpoint: chaves inesperadas: %s", unexpected)
