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
from experiments.training.core.ablation import run_ablation_demographics, run_pooled_behrt
from experiments.training.core.baselines import run_baseline_rf
from experiments.training.core.dataloaders import prepare_dataloaders, prepare_dataloaders_from_db, create_synthetic_client
from experiments.training.core.fl_core import (
    aggregate_fedavg,
    evaluate_global_model,
    run_federated_learning,
    run_federated_learning_manual,
    run_federated_learning_ray,
)
from experiments.training.core.orchestrator import FederatedTraining
from experiments.training.core.rag import run_rag_pipeline

__all__ = [
    "prepare_dataloaders",
    "prepare_dataloaders_from_db",
    "create_synthetic_client",
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
