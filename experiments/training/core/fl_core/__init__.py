"""
fl_core — Núcleo do aprendizado federado: agregação, avaliação global e loops de treinamento.

  aggregation.py   → aggregate_fedavg, aggregate_fednova, apply_dp_noise
  evaluation.py     → evaluate_global_model
  manual_loop.py     → run_federated_learning_manual (FL sequencial sem Ray)
  ray_loop.py          → run_federated_learning_ray (FL paralelo com Ray/Flower simulation)
  router.py             → run_federated_learning (roteador manual↔Ray baseado em USE_RAY)

Algoritmo de agregação selecionado por FED_CFG.use_fednova (config.py).
"""
from .aggregation import aggregate_fedavg, aggregate_fednova, apply_dp_noise
from .evaluation import evaluate_global_model
from .manual_loop import run_federated_learning_manual
from .ray_loop import run_federated_learning_ray
from .router import run_federated_learning

__all__ = [
    "aggregate_fedavg",
    "aggregate_fednova",
    "apply_dp_noise",
    "evaluate_global_model",
    "run_federated_learning_manual",
    "run_federated_learning_ray",
    "run_federated_learning",
]
