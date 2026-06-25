"""
experiments/training — submódulos do pipeline MOSAIC-FL.

  dataloaders  → prepare_dataloaders, prepare_dataloaders_from_db
  fl_core      → aggregate_fedavg, evaluate_global_model, run_federated_learning*
  rag          → run_rag_pipeline
  baselines    → run_baseline_rf
  ablation     → run_ablation_demographics, run_pooled_behrt
  orchestrator → FederatedTraining
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
