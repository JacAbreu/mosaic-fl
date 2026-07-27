"""
Testes para aggregate_calibration() e sua integração em weighted_average_evaluate_metrics()
— a etapa de agregação server-side da calibração federada (client-side fit em
FedProxClient._fit_local_calibrator, servidor só combina estatísticas comprimidas/
agregadas, nunca dado bruto por amostra).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import aggregate_calibration, weighted_average_evaluate_metrics


class TestAggregateCalibrationNone:
    def test_returns_none_when_no_client_sent_calibration(self):
        metrics = [(100, {"accuracy": 0.8}), (200, {"accuracy": 0.9})]
        assert aggregate_calibration(metrics) is None

    def test_returns_none_for_empty_list(self):
        assert aggregate_calibration([]) is None


class TestAggregateCalibrationTemperature:
    def test_weighted_average_of_temperature(self):
        metrics = [
            (100, {"calibration_method": "temperature", "temperature": 1.0}),
            (300, {"calibration_method": "temperature", "temperature": 2.0}),
        ]
        result = aggregate_calibration(metrics)
        assert result["calibration_method"] == "temperature"
        expected = (100 * 1.0 + 300 * 2.0) / 400
        assert abs(result["temperature"] - expected) < 1e-6

    def test_single_client_temperature(self):
        metrics = [(50, {"calibration_method": "temperature", "temperature": 1.5})]
        result = aggregate_calibration(metrics)
        assert abs(result["temperature"] - 1.5) < 1e-6


class TestAggregateCalibrationIsotonic:
    def test_pools_thresholds_across_clients_per_class(self):
        # 2 classes, 2 clientes
        thresholds_a = [[[0.1, 0.5], [0.0, 1.0]], [[0.2, 0.6], [0.0, 1.0]]]
        thresholds_b = [[[0.3, 0.9], [0.0, 1.0]], [[0.4, 0.8], [1.0, 0.0]]]
        metrics = [
            (100, {"calibration_method": "isotonic", "isotonic_thresholds_json": json.dumps(thresholds_a)}),
            (150, {"calibration_method": "isotonic", "isotonic_thresholds_json": json.dumps(thresholds_b)}),
        ]
        result = aggregate_calibration(metrics)
        assert result["calibration_method"] == "isotonic"
        assert result["isotonic_num_classes"] == 2

        pooled = json.loads(result["isotonic_pooled_thresholds_json"])
        assert len(pooled) == 2
        # classe 0: pontos dos dois clientes concatenados
        x0, y0 = pooled[0]
        assert set(x0) == {0.1, 0.5, 0.3, 0.9}
        assert len(x0) == len(y0) == 4

    def test_missing_thresholds_key_skipped_gracefully(self):
        metrics = [(100, {"calibration_method": "isotonic"})]
        assert aggregate_calibration(metrics) is None


class TestAggregateCalibrationIntegration:
    def test_integrated_into_weighted_average_evaluate_metrics(self):
        """Confirma que aggregate_calibration() é chamado dentro da função de
        agregação já registrada como evaluate_metrics_aggregation_fn (superlink.py)."""
        metrics = [
            (100, {
                "accuracy": 0.8, "f1_macro": 0.5,
                "calibration_method": "temperature", "temperature": 1.2,
            }),
        ]
        result = weighted_average_evaluate_metrics(metrics)
        assert result["calibration_method"] == "temperature"
        assert abs(result["temperature"] - 1.2) < 1e-6
        # métricas normais continuam presentes, sem regressão
        assert result["accuracy"] == 0.8

    def test_normal_round_without_calibration_unaffected(self):
        """Rodada sem calibrate=True no cliente — sem chaves de calibração, sem quebrar."""
        metrics = [(100, {"accuracy": 0.8, "f1_macro": 0.5})]
        result = weighted_average_evaluate_metrics(metrics)
        assert "calibration_method" not in result


class TestAggregateCalibrationDisagreement:
    """calibration_method="auto" (client.py::_fit_local_calibrator) deixa cada
    hospital escolher individualmente — achado 2026-07-26: o critério antigo
    ("primeiro cliente da lista decide pra todo mundo") descartava silenciosamente
    a contribuição de quem discordasse. Agora é maioria ponderada por n_samples,
    com log explícito quando há divergência."""

    def test_majority_by_sample_weight_wins_temperature(self):
        metrics = [
            (300, {"calibration_method": "temperature", "temperature": 1.5}),
            (50, {"calibration_method": "isotonic",
                  "isotonic_thresholds_json": json.dumps([[[0.1], [0.0]]])}),
        ]
        result = aggregate_calibration(metrics)
        assert result["calibration_method"] == "temperature"
        assert abs(result["temperature"] - 1.5) < 1e-6

    def test_majority_by_sample_weight_wins_isotonic(self):
        thresholds = [[[0.1, 0.5], [0.0, 1.0]]]
        metrics = [
            (50, {"calibration_method": "temperature", "temperature": 1.5}),
            (300, {"calibration_method": "isotonic",
                   "isotonic_thresholds_json": json.dumps(thresholds)}),
        ]
        result = aggregate_calibration(metrics)
        assert result["calibration_method"] == "isotonic"

    def test_minority_method_excluded_from_temperature_average(self):
        """Documenta o comportamento atual: quando há divergência, só quem ficou
        do lado majoritário entra na média — o outro hospital não contribui pro
        calibrador federado desta rodada (limitação conhecida, registrada em log,
        não mais um acidente de ordenação de lista)."""
        metrics = [
            (100, {"calibration_method": "temperature", "temperature": 1.0}),
            (200, {"calibration_method": "temperature", "temperature": 3.0}),
            (50, {"calibration_method": "isotonic",
                  "isotonic_thresholds_json": json.dumps([[[0.1], [0.0]]])}),
        ]
        result = aggregate_calibration(metrics)
        assert result["calibration_method"] == "temperature"
        expected = (100 * 1.0 + 200 * 3.0) / 300
        assert abs(result["temperature"] - expected) < 1e-6

    def test_logs_warning_on_disagreement(self, caplog):
        metrics = [
            (300, {"calibration_method": "temperature", "temperature": 1.5}),
            (50, {"calibration_method": "isotonic",
                  "isotonic_thresholds_json": json.dumps([[[0.1], [0.0]]])}),
        ]
        with caplog.at_level("WARNING", logger="mosaicfl.core.federated"):
            aggregate_calibration(metrics)
        assert any("calibration_method_disagreement" in r.message for r in caplog.records)

    def test_no_warning_when_all_agree(self, caplog):
        metrics = [
            (100, {"calibration_method": "temperature", "temperature": 1.0}),
            (200, {"calibration_method": "temperature", "temperature": 2.0}),
        ]
        with caplog.at_level("WARNING", logger="mosaicfl.core.federated"):
            aggregate_calibration(metrics)
        assert not any("calibration_method_disagreement" in r.message for r in caplog.records)
