"""
Testes para ProductionFedProxStrategy._save_federated_training_id_marker() — marca
o training_id concluído como "modelo ativo" (CheckpointStore.mark_active_model),
pra API descobrir automaticamente qual carregar. Substitui o mecanismo anterior de
arquivo (experiments/last_federated_training_id.txt) — achado 2026-07-25/26:
arquivo local não é fonte de verdade compartilhada entre processos/máquinas físicas
diferentes (ver migration 024_fl_trainings_active_model).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(training_id=99):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy._training_id = training_id
    strategy._checkpoint_store = MagicMock()
    return strategy


class TestSaveFederatedTrainingIdMarker:
    def test_calls_mark_active_model_with_training_id(self):
        strategy = _make_strategy(training_id=70)

        strategy._save_federated_training_id_marker()

        strategy._checkpoint_store.mark_active_model.assert_called_once_with(70)

    def test_never_raises_when_checkpoint_store_fails(self):
        """Nunca propaga exceção — é um atalho de conveniência, não uma dependência
        crítica do treino (sem isso, a API só cai no fallback de maior accuracy global)."""
        strategy = _make_strategy(training_id=1)
        strategy._checkpoint_store.mark_active_model.side_effect = RuntimeError("banco fora do ar")

        strategy._save_federated_training_id_marker()  # não deve levantar exceção

    def test_uses_current_training_id_not_stale_value(self):
        strategy = _make_strategy(training_id=123)

        strategy._save_federated_training_id_marker()

        strategy._checkpoint_store.mark_active_model.assert_called_once_with(123)
