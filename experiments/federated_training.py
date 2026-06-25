"""
federated_training.py — shim de compatibilidade.

A lógica foi dividida em submódulos sob experiments/training/:
  dataloaders  → prepare_dataloaders, prepare_dataloaders_from_db
  fl_core      → aggregate_fedavg, evaluate_global_model, run_federated_learning*
  rag          → run_rag_pipeline
  baselines    → run_baseline_rf
  ablation     → run_ablation_demographics, run_pooled_behrt
  orchestrator → FederatedTraining

Importar deste módulo continua funcionando — nenhum script externo precisa mudar.
"""
from experiments.training.ablation import run_ablation_demographics, run_pooled_behrt
from experiments.training.baselines import run_baseline_rf
from experiments.training.dataloaders import prepare_dataloaders, prepare_dataloaders_from_db
from experiments.training.fl_core import (
    aggregate_fedavg,
    evaluate_global_model,
    run_federated_learning,
    run_federated_learning_manual,
    run_federated_learning_ray,
)
from experiments.training.orchestrator import FederatedTraining
from experiments.training.rag import run_rag_pipeline

__all__ = [
    "prepare_dataloaders",
    "prepare_dataloaders_from_db",
    "aggregate_fedavg",
    "evaluate_global_model",
    "run_federated_learning_manual",
    "run_federated_learning_ray",
    "run_federated_learning",
    "run_rag_pipeline",
    "run_baseline_rf",
    "run_ablation_demographics",
    "run_pooled_behrt",
    "FederatedTraining",
]
