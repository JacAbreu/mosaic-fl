"""
strategy — Estratégia FedProx de produção (mosaicfl.v2) com checkpoint e convergência.

ProductionFedProxStrategy é composta via mixins — a API pública (todos os métodos)
é idêntica à versão anterior de arquivo único:
  core.py                — __init__, aggregate_fit, aggregate_evaluate, _save_evaluation_report
                            (+ CHECKPOINT_DIR, LOG_DIR — ver nota de patch-target no arquivo)
  fit_config_mixin.py       — configure_fit, _load_global_weights
  watchdog_mixin.py           — _restore_from_state, _start_round_watchdog, _cancel_round_watchdog
  calibration_mixin.py           — _run_calibration
"""
from mosaicfl.core.convergence import ConvergenceTracker

from .core import CHECKPOINT_DIR, LOG_DIR, ProductionFedProxStrategy

__all__ = ["ProductionFedProxStrategy", "ConvergenceTracker", "CHECKPOINT_DIR", "LOG_DIR"]
