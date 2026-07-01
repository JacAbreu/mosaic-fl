"""
experiments/training_runner — entrypoints executáveis do pipeline MOSAIC-FL.

Cada arquivo é invocado como script (`python experiments/training_runner/run_X.py`),
não importado como módulo. Ver Makefile para os targets que os invocam.

  run_training.py                → treinamento federado com dados reais FAPESP
  run_experiments_simulation.py   → simulação com dados sintéticos
  run_behrt_pooled.py             → baseline BEHRT pooled (artefato de pesquisa)
  run_recalibrate.py              → recalibração pós-treino (temperature/isotônica)
  run_bootstrap_ci.py             → intervalo de confiança via bootstrap
  run_seed_sensitivity.py         → análise de sensibilidade a múltiplos seeds
  run_federated_real.py           → rede federada real (servidor/cliente via socket)
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
