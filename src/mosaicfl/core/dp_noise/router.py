"""
router.py — Escolhe a estratégia de ruído DP a partir de FED_CFG.dp_noise_strategy
("uniform", padrão | "layer_group", opcional). Mesmo padrão de roteamento de
mosaicfl.core.class_weighting.router.
"""
from .base import DPNoiseStrategy
from .layer_group import LayerGroupNoiseStrategy
from .uniform import UniformNoiseStrategy


def get_dp_noise_strategy() -> DPNoiseStrategy:
    from ..config import FED_CFG

    if FED_CFG.dp_noise_strategy == "layer_group":
        return LayerGroupNoiseStrategy(
            head_scale=FED_CFG.dp_noise_head_scale,
            embedding_scale=FED_CFG.dp_noise_embedding_scale,
            transformer_scale=FED_CFG.dp_noise_transformer_scale,
        )
    return UniformNoiseStrategy()
