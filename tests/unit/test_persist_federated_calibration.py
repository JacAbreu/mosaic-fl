"""
Testes para ProductionFedProxStrategy._persist_federated_calibration() — a etapa que
recebe o resultado já agregado (aggregate_calibration em federated.py) e persiste no
checkpoint via CheckpointStore, para a InferenceEngine carregar depois.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy():
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy.global_model = SimplifiedBEHRT()
    strategy.vocab = {"A": 2}
    strategy._training_id = 7
    strategy._checkpoint_store = MagicMock()
    strategy._last_round_metrics = {"accuracy": 0.7, "loss": 0.3}
    # Pesos/round/accuracy da "melhor rodada" (achado 2026-07-28) — usados por
    # _persist_federated_calibration() em vez dos pesos da rodada atual.
    strategy._best_state_dict = None  # cai no fallback self.global_model.state_dict()
    strategy._best_confusion_matrix = None
    strategy._best_round = 7
    strategy._best_accuracy = 0.7
    # Usados por _build_evaluation_json() (achado #6 da auditoria Caminho A vs B).
    strategy.round_counter = 10
    strategy._best_f1_macro = 0.5
    strategy._best_macro_auc = 0.8
    strategy._ece = None
    strategy._ece_pre = None
    strategy._dp_epsilon_simple = None
    strategy._rdp_accountant = None
    return strategy


class TestPersistFederatedCalibrationUsesBestRoundPerClassF1:
    """Regressão do bug achado 2026-08-08 (training_id=6/8, real): a chamada
    salvava state_dict/accuracy da MELHOR rodada mas per_class_f1 vinha de
    aggregated_metrics — a rodada ATUAL (onde calibração roda, normalmente a
    última/de early stop), reintroduzindo o mesmo bug de checkpoint corrigido
    em 2026-07-28, só que pra este campo específico."""

    def test_evaluation_json_usa_best_per_class_f1_nao_o_da_rodada_atual(self):
        strategy = _make_strategy()
        strategy._best_per_class_f1 = [0.9, 0.8, 0.7, 0.6, 0.5]  # rodada 7, a melhor
        # aggregated_metrics simula a rodada ATUAL (ex.: 47, colapsada) — bem
        # diferente da melhor. Se o bug reaparecer, é isto que vai ser salvo.
        rodada_atual_diferente = [0.0, 0.0, 0.0, 0.0, 0.1]
        strategy._persist_federated_calibration(
            server_round=47,
            calibration_method="temperature",
            aggregated_metrics={
                "temperature": 1.0,
                "per_class_f1_json": json.dumps(rodada_atual_diferente),
            },
        )
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["evaluation_json"]["best_per_class_f1"] == [0.9, 0.8, 0.7, 0.6, 0.5]
        assert kwargs["evaluation_json"]["best_per_class_f1"] != rodada_atual_diferente

    def test_fallback_quando_best_per_class_f1_nunca_foi_setado(self):
        """Cenário legado/recovery: se _best_per_class_f1 nunca foi capturado nesta
        run (None), cai de volta em aggregated_metrics — não trava a calibração."""
        strategy = _make_strategy()
        strategy._best_per_class_f1 = None
        fallback = [0.4, 0.3, 0.2, 0.1, 0.0]
        strategy._persist_federated_calibration(
            server_round=47,
            calibration_method="temperature",
            aggregated_metrics={"temperature": 1.0, "per_class_f1_json": json.dumps(fallback)},
        )
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["evaluation_json"]["best_per_class_f1"] == fallback


class TestPersistFederatedCalibrationTemperature:
    def test_saves_temperature_to_checkpoint_store(self):
        strategy = _make_strategy()
        strategy._persist_federated_calibration(
            server_round=10,
            calibration_method="temperature",
            aggregated_metrics={"calibration_method": "temperature", "temperature": 1.8},
        )
        strategy._checkpoint_store.save.assert_called_once()
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["calibration_method"] == "temperature"
        assert kwargs["temperature"] == 1.8
        assert kwargs["isotonic_calibrators"] is None
        assert kwargs["training_id"] == 7


class TestPersistFederatedCalibrationIsotonic:
    def test_rebuilds_and_saves_isotonic_calibrators(self):
        import json
        strategy = _make_strategy()
        pooled = [[[0.1, 0.5, 0.9], [0.0, 0.5, 1.0]], [[0.2, 0.6], [0.0, 1.0]]]
        strategy._persist_federated_calibration(
            server_round=10,
            calibration_method="isotonic",
            aggregated_metrics={
                "calibration_method": "isotonic",
                "isotonic_pooled_thresholds_json": json.dumps(pooled),
                "isotonic_num_classes": 2,
            },
        )
        strategy._checkpoint_store.save.assert_called_once()
        kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert kwargs["calibration_method"] == "isotonic"
        assert kwargs["isotonic_num_classes"] == 2
        assert len(kwargs["isotonic_calibrators"]) == 2

    def test_incomplete_isotonic_data_skips_save(self):
        strategy = _make_strategy()
        strategy._persist_federated_calibration(
            server_round=10,
            calibration_method="isotonic",
            aggregated_metrics={"calibration_method": "isotonic"},  # sem thresholds
        )
        strategy._checkpoint_store.save.assert_not_called()


class TestPersistFederatedCalibrationRobustness:
    def test_no_checkpoint_store_does_not_raise(self):
        strategy = _make_strategy()
        strategy._checkpoint_store = None
        strategy._persist_federated_calibration(
            server_round=10, calibration_method="temperature",
            aggregated_metrics={"temperature": 1.0},
        )  # não deve levantar exceção

    def test_unknown_method_skips_save(self):
        strategy = _make_strategy()
        strategy._persist_federated_calibration(
            server_round=10, calibration_method="bogus", aggregated_metrics={},
        )
        strategy._checkpoint_store.save.assert_not_called()

    def test_exception_during_save_does_not_propagate(self):
        """Calibração é enriquecimento pós-convergência — uma falha aqui não deve
        derrubar o restante de aggregate_evaluate()."""
        strategy = _make_strategy()
        strategy._checkpoint_store.save.side_effect = RuntimeError("banco fora do ar")
        strategy._persist_federated_calibration(
            server_round=10, calibration_method="temperature",
            aggregated_metrics={"temperature": 1.0},
        )  # não deve levantar exceção
