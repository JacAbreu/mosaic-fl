"""
Testes para aggregate_resource_metrics() e sua integração em weighted_average_loss()
— agregação server-side do custo computacional por rodada (FedProxClient.fit(), só
presente quando FL_COLLECT_RESOURCE_METRICS=1). Energia da GPU é somada entre
clientes (hardware físico distinto por cliente), não uma média ponderada.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import aggregate_resource_metrics, weighted_average_loss


class TestAggregateResourceMetricsNone:
    def test_returns_none_when_no_client_sent_resources(self):
        metrics = [(100, {"loss": 0.5}), (200, {"loss": 0.4})]
        assert aggregate_resource_metrics(metrics) is None

    def test_returns_none_for_empty_list(self):
        assert aggregate_resource_metrics([]) is None


class TestAggregateResourceMetricsCpuOnly:
    def test_per_client_breakdown_without_gpu(self):
        metrics = [
            (100, {"resource_duration_s": 60.0, "resource_cpu_pct": 400.0, "resource_ram_mb": 500.0}),
            (200, {"resource_duration_s": 90.0, "resource_cpu_pct": 300.0, "resource_ram_mb": 700.0}),
        ]
        result = aggregate_resource_metrics(metrics)
        assert result is not None
        assert "resource_round_gpu_energy_wh" not in result
        assert "resource_round_gpu_avg_power_w" not in result
        per_client = json.loads(result["resource_per_client_json"])
        assert len(per_client) == 2
        assert per_client[0]["duration_s"] == 60.0
        assert per_client[1]["cpu_pct"] == 300.0

    def test_partial_gpu_only_one_client_has_gpu(self):
        """1 cliente com GPU (BPSP desktop), 1 sem (HSL notebook) — cenário real do projeto."""
        metrics = [
            (100, {
                "resource_duration_s": 60.0, "resource_cpu_pct": 800.0, "resource_ram_mb": 500.0,
                "resource_gpu_power_w": 150.0, "resource_gpu_energy_wh": 2.5,
            }),
            (200, {"resource_duration_s": 90.0, "resource_cpu_pct": 300.0, "resource_ram_mb": 700.0}),
        ]
        result = aggregate_resource_metrics(metrics)
        assert result["resource_round_gpu_energy_wh"] == 2.5
        assert result["resource_round_gpu_avg_power_w"] == 150.0
        assert result["resource_round_gpu_peak_power_w"] == 150.0
        per_client = json.loads(result["resource_per_client_json"])
        assert "gpu_power_w" in per_client[0]
        assert "gpu_power_w" not in per_client[1]


class TestAggregateResourceMetricsGpuSum:
    def test_gpu_energy_summed_not_averaged(self):
        metrics = [
            (100, {
                "resource_duration_s": 60.0, "resource_cpu_pct": 800.0, "resource_ram_mb": 500.0,
                "resource_gpu_power_w": 150.0, "resource_gpu_energy_wh": 2.5,
            }),
            (100, {
                "resource_duration_s": 60.0, "resource_cpu_pct": 800.0, "resource_ram_mb": 500.0,
                "resource_gpu_power_w": 50.0, "resource_gpu_energy_wh": 0.8,
            }),
        ]
        result = aggregate_resource_metrics(metrics)
        assert abs(result["resource_round_gpu_energy_wh"] - 3.3) < 1e-9
        assert result["resource_round_gpu_avg_power_w"] == 100.0
        assert result["resource_round_gpu_peak_power_w"] == 150.0


class TestWeightedAverageLossIntegration:
    def test_loss_still_computed_alongside_resources(self):
        metrics = [
            (100, {"loss": 0.5, "resource_duration_s": 60.0, "resource_cpu_pct": 400.0, "resource_ram_mb": 500.0}),
            (100, {"loss": 0.3, "resource_duration_s": 60.0, "resource_cpu_pct": 400.0, "resource_ram_mb": 500.0}),
        ]
        result = weighted_average_loss(metrics)
        assert abs(result["loss"] - 0.4) < 1e-9
        assert "resource_per_client_json" in result

    def test_loss_unaffected_when_no_resource_data(self):
        metrics = [(100, {"loss": 0.5}), (100, {"loss": 0.3})]
        result = weighted_average_loss(metrics)
        assert abs(result["loss"] - 0.4) < 1e-9
        assert "resource_per_client_json" not in result


class TestAggregateResourceMetricsRamCpu:
    """peak_ram_mb/avg_cpu_pct de fl_trainings ficavam sempre em 0.0 no Caminho B —
    a captura por cliente existia mas nunca era resumida em nível de rodada. Achado
    2026-07-26, mesma classe de lacuna já corrigida pra energia da GPU."""

    def test_peak_ram_is_max_not_average(self):
        metrics = [
            (100, {"resource_duration_s": 60.0, "resource_cpu_pct": 200.0, "resource_ram_mb": 500.0}),
            (100, {"resource_duration_s": 60.0, "resource_cpu_pct": 400.0, "resource_ram_mb": 1200.0}),
        ]
        result = aggregate_resource_metrics(metrics)
        assert result["resource_round_peak_ram_mb"] == 1200.0

    def test_avg_cpu_is_mean_of_samples(self):
        metrics = [
            (100, {"resource_duration_s": 60.0, "resource_cpu_pct": 200.0, "resource_ram_mb": 500.0}),
            (100, {"resource_duration_s": 60.0, "resource_cpu_pct": 400.0, "resource_ram_mb": 500.0}),
        ]
        result = aggregate_resource_metrics(metrics)
        assert result["resource_round_avg_cpu_pct"] == 300.0

    def test_present_even_without_gpu(self):
        """RAM/CPU são amostrados independente de GPU existir — não devem ficar
        condicionados à presença de resource_gpu_power_w."""
        metrics = [(100, {"resource_duration_s": 60.0, "resource_cpu_pct": 250.0, "resource_ram_mb": 600.0})]
        result = aggregate_resource_metrics(metrics)
        assert "resource_round_gpu_energy_wh" not in result
        assert result["resource_round_peak_ram_mb"] == 600.0
        assert result["resource_round_avg_cpu_pct"] == 250.0
