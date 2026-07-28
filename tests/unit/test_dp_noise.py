"""
Testes para mosaicfl.core.dp_noise — Strategy pattern de ruído DP-FedAvg (achado
2026-07-28: "o que tem no Caminho A tem que ter no Caminho B", decisão da autora).
UniformNoiseStrategy (padrão, extração exata do apply_dp_noise histórico) e
LayerGroupNoiseStrategy (opcional, ruído diferenciado por grupo de camada — ver
docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14/7.9).
"""
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.dp_noise import (
    LayerGroupNoiseStrategy,
    UniformNoiseStrategy,
    apply_dp_noise,
)


class TestUniformNoiseStrategy:
    def test_same_multiplier_for_any_key(self):
        strategy = UniformNoiseStrategy()
        assert strategy.multiplier_for_group(strategy.group_for_key("embedding.weight"), 0.5) == 0.5
        assert strategy.multiplier_for_group(strategy.group_for_key("classifier.3.bias"), 0.5) == 0.5

    def test_single_group(self):
        strategy = UniformNoiseStrategy()
        multipliers = strategy.group_multipliers(
            ["embedding.weight", "layers.0.norm1.weight", "classifier.3.bias"], 0.3,
        )
        assert multipliers == {"all": 0.3}


class TestLayerGroupNoiseStrategy:
    def test_classifies_known_keys_correctly(self):
        strategy = LayerGroupNoiseStrategy()
        assert strategy.group_for_key("embedding.weight") == "embedding"
        assert strategy.group_for_key("dia_embedding.embedding.weight") == "embedding"
        assert strategy.group_for_key("cls_token") == "embedding"
        assert strategy.group_for_key("layers.0.self_attn.in_proj_weight") == "transformer"
        assert strategy.group_for_key("layers.1.norm2.bias") == "transformer"
        assert strategy.group_for_key("pre_classifier.0.weight") == "head"
        assert strategy.group_for_key("classifier.3.bias") == "head"
        assert strategy.group_for_key("pos_encoder.pe") == "excluded"

    def test_unknown_key_falls_back_to_other_with_base_multiplier(self):
        strategy = LayerGroupNoiseStrategy(head_scale=0.5)
        group = strategy.group_for_key("algo_novo_nao_previsto.weight")
        assert group == "other"
        assert strategy.multiplier_for_group(group, 1.0) == 1.0

    def test_excluded_group_gets_zero_multiplier(self):
        strategy = LayerGroupNoiseStrategy()
        assert strategy.multiplier_for_group("excluded", 0.5) == 0.0

    def test_head_scale_reduces_noise_on_classifier(self):
        strategy = LayerGroupNoiseStrategy(head_scale=0.5)
        assert strategy.multiplier_for_group("head", 1.0) == pytest.approx(0.5)
        assert strategy.multiplier_for_group("transformer", 1.0) == pytest.approx(1.0)
        assert strategy.multiplier_for_group("embedding", 1.0) == pytest.approx(1.0)

    def test_default_scales_are_noop_matching_uniform(self):
        """Sem configurar escalas explicitamente, layer_group deve produzir o MESMO
        multiplicador que uniform em qualquer grupo protegido — só difere quando
        alguém deliberadamente configura uma escala != 1.0."""
        strategy = LayerGroupNoiseStrategy()
        for group in ("head", "transformer", "embedding", "other"):
            assert strategy.multiplier_for_group(group, 0.7) == pytest.approx(0.7)

    def test_group_multipliers_covers_all_distinct_groups(self):
        strategy = LayerGroupNoiseStrategy(head_scale=0.5, embedding_scale=2.0)
        keys = [
            "embedding.weight", "cls_token", "layers.0.norm1.weight",
            "pre_classifier.0.weight", "classifier.3.bias", "pos_encoder.pe",
        ]
        multipliers = strategy.group_multipliers(keys, 1.0)
        assert multipliers == {
            "embedding": 2.0, "transformer": 1.0, "head": 0.5, "excluded": 0.0,
        }


class TestApplyDpNoise:
    def _state_dict(self):
        return OrderedDict({
            "embedding.weight": torch.zeros(4, 4),
            "layers.0.norm1.weight": torch.zeros(4),
            "classifier.3.bias": torch.zeros(5),
            "pos_encoder.pe": torch.zeros(1, 8, 4),
        })

    def test_uniform_strategy_adds_noise_to_all_keys(self):
        state = self._state_dict()
        eps, groups = apply_dp_noise(
            state, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0,
            strategy=UniformNoiseStrategy(),
        )
        assert groups == {"all": 1.0}
        assert eps > 0
        # ruído gaussiano com std>0 quase certamente não deixa o tensor exatamente zero
        assert not torch.allclose(state["pos_encoder.pe"], torch.zeros(1, 8, 4))

    def test_layer_group_strategy_excludes_pos_encoder(self):
        state = self._state_dict()
        apply_dp_noise(
            state, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0,
            strategy=LayerGroupNoiseStrategy(),
        )
        # excluído: continua exatamente zero, sem ruído nenhum
        assert torch.allclose(state["pos_encoder.pe"], torch.zeros(1, 8, 4))
        # protegido: recebe ruído normalmente
        assert not torch.allclose(state["embedding.weight"], torch.zeros(4, 4))

    def test_no_strategy_defaults_to_uniform(self):
        state = self._state_dict()
        eps, groups = apply_dp_noise(
            state, round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0,
        )
        assert groups == {"all": 1.0}

    def test_epsilon_uses_weakest_multiplier_among_protected_groups(self):
        """head_scale=0.1 é o multiplicador MAIS BAIXO entre os grupos protegidos —
        epsilon simples precisa refletir o pior caso (menor multiplicador = mais
        ruído = MENOR epsilon; mas a fórmula é 1/multiplicador, então multiplicador
        baixo -> epsilon ALTO — o pior caso de privacidade é o de MENOR proteção,
        ou seja, o multiplicador mais alto entre os grupos. Testa que a função usa
        min(multipliers) na fórmula, que corresponde ao MAIOR epsilon possível)."""
        state = self._state_dict()
        eps_uniform, _ = apply_dp_noise(
            OrderedDict({k: v.clone() for k, v in state.items()}),
            round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0,
            strategy=UniformNoiseStrategy(),
        )
        eps_layer, groups = apply_dp_noise(
            OrderedDict({k: v.clone() for k, v in state.items()}),
            round_num=1, n_clients=2, noise_multiplier=1.0, max_grad_norm=1.0,
            strategy=LayerGroupNoiseStrategy(head_scale=0.1),
        )
        # grupo "head" tem multiplicador efetivo 0.1 (mais baixo que 1.0 dos demais)
        # -> epsilon usa esse valor -> epsilon MAIOR que o cenário uniforme (pior privacidade)
        assert groups["head"] == pytest.approx(0.1)
        assert eps_layer > eps_uniform

    def test_all_groups_excluded_falls_back_to_base_multiplier(self):
        """Caso degenerado (não deveria acontecer na prática, mas não pode quebrar):
        se todo grupo presente tiver multiplicador 0, usa noise_multiplier base pra
        não dividir por zero."""
        class _AllExcluded(UniformNoiseStrategy):
            def multiplier_for_group(self, group, base_noise_multiplier):
                return 0.0

        state = OrderedDict({"x": torch.zeros(3)})
        eps, groups = apply_dp_noise(
            state, round_num=1, n_clients=2, noise_multiplier=0.8, max_grad_norm=1.0,
            strategy=_AllExcluded(),
        )
        assert eps > 0  # não levanta ZeroDivisionError
