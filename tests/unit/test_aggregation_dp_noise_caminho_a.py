"""
Testes para experiments/training/core/fl_core/aggregation.py::apply_dp_noise —
achado 2026-07-28: função virou um wrapper fino sobre mosaicfl.core.dp_noise
(Strategy pattern), mas precisa manter a assinatura E o comportamento histórico
exatos (Caminho A não pode quebrar) quando FED_CFG.dp_noise_strategy="uniform"
(padrão). Primeiro teste de DP do Caminho A.
"""
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "training" / "core"))

from fl_core.aggregation import apply_dp_noise


def _state_dict():
    return OrderedDict({
        "embedding.weight": torch.zeros(4, 4),
        "layers.0.norm1.weight": torch.zeros(4),
        "classifier.3.bias": torch.zeros(5),
    })


class TestApplyDpNoiseCaminhoA:
    def test_signature_and_return_type_unchanged(self):
        """Assinatura E tipo de retorno (float) precisam ser idênticos ao histórico
        — manual_loop.py chama isso posicionalmente, sem keyword args novos."""
        state = _state_dict()
        result = apply_dp_noise(state, 1, 2, 1.0, 1.0)
        assert isinstance(result, float)
        assert result > 0

    def test_default_strategy_is_uniform_adds_noise_everywhere(self, monkeypatch):
        import mosaicfl.core.config as config_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(config_module, "FED_CFG", FedConfig(dp_noise_strategy="uniform"))

        state = _state_dict()
        apply_dp_noise(state, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0)

        assert not torch.allclose(state["embedding.weight"], torch.zeros(4, 4))
        assert not torch.allclose(state["classifier.3.bias"], torch.zeros(5))

    def test_layer_group_strategy_selectable_via_config(self, monkeypatch):
        """Caminho A também ganha a estratégia nova via FED_CFG — 'o que tem no
        Caminho A tem que ter no Caminho B' vale nos dois sentidos."""
        import mosaicfl.core.config as config_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(config_module, "FED_CFG", FedConfig(
            dp_noise_strategy="layer_group",
            dp_noise_head_scale=0.0,
            dp_noise_transformer_scale=1.0,
            dp_noise_embedding_scale=1.0,
        ))

        state = _state_dict()
        apply_dp_noise(state, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0)

        # head_scale=0.0 -> classifier.* não recebe ruído nenhum
        assert torch.allclose(state["classifier.3.bias"], torch.zeros(5))
        # embedding/transformer continuam recebendo ruído normalmente
        assert not torch.allclose(state["embedding.weight"], torch.zeros(4, 4))

    def test_round_num_scales_accumulated_epsilon(self):
        state1 = _state_dict()
        eps_round1 = apply_dp_noise(state1, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0)
        state5 = _state_dict()
        eps_round5 = apply_dp_noise(state5, round_num=5, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0)
        assert eps_round5 == pytest.approx(eps_round1 * 5)
