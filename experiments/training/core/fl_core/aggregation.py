"""
aggregation.py — Agregação de state_dicts (FedAvg, FedNova) e ruído DP.

  aggregate_fedavg  — média ponderada de state_dicts (FedAvg)
  aggregate_fednova — média ponderada normalizada por passos efetivos τ_i (FedNova)
  apply_dp_noise    — ruído gaussiano DP-FedAvg (McMahan et al. 2018)

Algoritmo de agregação selecionado por FED_CFG.use_fednova (config.py).
"""
import logging
import math
from collections import OrderedDict
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def aggregate_fedavg(state_dicts: List[OrderedDict], weights: List[int]) -> OrderedDict:
    """Agrega state_dicts via média ponderada (FedAvg)."""
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Peso total de agregação é zero.")

    global_state = OrderedDict()
    for key in state_dicts[0].keys():
        global_state[key] = torch.zeros_like(state_dicts[0][key].float())

    for state_dict, weight in zip(state_dicts, weights):
        for key in state_dict.keys():
            global_state[key] += state_dict[key].float() * (weight / total_weight)

    return global_state


def aggregate_fednova(
    global_state: OrderedDict,
    client_states: List[OrderedDict],
    weights: List[int],
    tau_values: List[int],
) -> Tuple[OrderedDict, float]:
    """Agrega state_dicts via FedNova — normaliza updates por passos efetivos τ_i.

    Corrige o viés de agregação em clientes heterogêneos: clientes com mais dados
    (mais batches por rodada) têm seus updates normalizados por τ_i antes de agregar,
    equalizando a contribuição independente do volume local.

    Fórmula (Wang et al. 2020):
        τ_eff = Σ p_i · τ_i
        w_{t+1} = w_t + τ_eff · Σ p_i · (w_i − w_t) / τ_i

    Referência: Wang et al. 2020 — "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization"
    """
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Peso total de agregação é zero.")

    tau_eff = sum(n * tau for n, tau in zip(weights, tau_values)) / total_weight

    new_state = OrderedDict()
    for key in global_state.keys():
        w_t = global_state[key].float()
        normalized_delta = torch.zeros_like(w_t)
        for state, n, tau in zip(client_states, weights, tau_values):
            p_i = n / total_weight
            normalized_delta += p_i * (state[key].float() - w_t) / max(tau, 1)
        new_state[key] = (w_t + tau_eff * normalized_delta).to(global_state[key].dtype)

    return new_state, tau_eff


def apply_dp_noise(
    global_state: OrderedDict,
    round_num: int,
    n_clients: int,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float = 1e-5,
) -> float:
    """Adiciona ruído gaussiano ao estado global agregado (DP-FedAvg, McMahan et al. 2018).

    noise_std = σ × S / n_clients
    ε por rodada ≈ √(2 ln(1.25/δ)) / σ   (mecanismo Gaussiano — cota superior)
    Para cotas mais apertadas, usar RDP/moments accountant (ex: Opacus).

    Retorna ε acumulado (composição simples × rodadas).
    """
    noise_std = noise_multiplier * max_grad_norm / max(n_clients, 1)
    with torch.no_grad():
        for key in global_state:
            noise = torch.normal(
                0.0, noise_std, size=global_state[key].shape, device=global_state[key].device
            )
            global_state[key] = (global_state[key].float() + noise).to(global_state[key].dtype)

    eps_per_round = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    eps_accumulated = eps_per_round * round_num
    logger.info(
        "dp_noise σ=%.2f S=%.2f noise_std=%.6f n=%d | "
        "ε_rodada≈%.3f ε_acum≈%.3f δ=%.0e (cota superior — composição simples)",
        noise_multiplier, max_grad_norm, noise_std, n_clients,
        eps_per_round, eps_accumulated, delta,
    )
    return eps_accumulated
