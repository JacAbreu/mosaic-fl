"""
experiments/training — orquestração e adapters de experimentação do MOSAIC-FL.

  core/                → mecânica do pipeline federado (dataloaders, fl_core, rag,
                          baselines, ablation, orchestrator) — ver core/__init__.py
  federated_training.py → shim de compatibilidade sobre core/ (FederatedTraining etc.)
  experiment_server.py  → adapter de servidor FL local (Flower) para experimentos
"""
