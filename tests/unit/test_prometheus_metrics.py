"""
Tests for Prometheus metrics exposition in HealthServer.

Exercises: /metrics returns valid Prometheus text, metrics update on
set_round_metrics, fl_convergence_round tracks convergence, push_metrics
skips gracefully when FL_PUSHGATEWAY_URL is absent.
"""
import time
import urllib.request
from unittest.mock import patch

import pytest

from infrastructure.shared.health_server import HealthServer
from infrastructure.shared.metrics import (
    REGISTRY,
    fl_rounds_total,
    fl_round_accuracy,
    fl_round_loss,
    fl_convergence_round,
)


def _get(port: int, path: str) -> tuple[int, str]:
    url = f"http://localhost:{port}{path}"
    with urllib.request.urlopen(url, timeout=3) as resp:
        return resp.status, resp.read().decode()


@pytest.fixture(scope="module")
def live_server():
    """Único servidor HTTP por módulo — evita conflito de porta entre testes."""
    hs = HealthServer(port=18081)
    hs.start()
    time.sleep(0.1)
    yield hs
    hs.stop()


class TestPrometheusEndpoint:
    """HealthServer deve expor /metrics no formato Prometheus text."""

    def test_metrics_returns_200(self, live_server):
        status, _ = _get(18081, "/metrics")
        assert status == 200

    def test_metrics_content_type_is_prometheus(self, live_server):
        url = "http://localhost:18081/metrics"
        with urllib.request.urlopen(url, timeout=3) as resp:
            ct = resp.headers.get("Content-Type", "")
        assert "text/plain" in ct or "openmetrics" in ct

    def test_metrics_body_contains_fl_rounds(self, live_server):
        _, body = _get(18081, "/metrics")
        assert "fl_rounds_total" in body

    def test_metrics_body_contains_fl_accuracy(self, live_server):
        _, body = _get(18081, "/metrics")
        assert "fl_round_accuracy" in body


class TestMetricsUpdatedBySetRoundMetrics:
    """set_round_metrics deve atualizar os Gauges do Prometheus."""

    def test_accuracy_gauge_updated(self):
        hs = HealthServer(port=0)  # sem HTTP
        before = fl_round_accuracy._value.get()
        hs.set_round_metrics(1, {"accuracy": 0.91, "loss": 0.12, "convergence_round": None})
        assert fl_round_accuracy._value.get() == pytest.approx(0.91)

    def test_loss_gauge_updated(self):
        hs = HealthServer(port=0)
        hs.set_round_metrics(2, {"accuracy": 0.80, "loss": 0.25, "convergence_round": None})
        assert fl_round_loss._value.get() == pytest.approx(0.25)

    def test_rounds_counter_increments(self):
        hs = HealthServer(port=0)
        before = fl_rounds_total._value.get()
        hs.set_round_metrics(3, {"accuracy": 0.75, "loss": 0.3, "convergence_round": None})
        assert fl_rounds_total._value.get() == before + 1

    def test_convergence_round_set_when_present(self):
        hs = HealthServer(port=0)
        hs.set_round_metrics(5, {"accuracy": 0.88, "loss": 0.1, "convergence_round": 5})
        assert fl_convergence_round._value.get() == pytest.approx(5)

    def test_convergence_round_unchanged_when_none(self):
        hs = HealthServer(port=0)
        fl_convergence_round.set(-1)
        hs.set_round_metrics(6, {"accuracy": 0.88, "loss": 0.1, "convergence_round": None})
        assert fl_convergence_round._value.get() == pytest.approx(-1)

    def test_none_accuracy_defaults_to_zero(self):
        hs = HealthServer(port=0)
        hs.set_round_metrics(7, {"accuracy": None, "loss": None, "convergence_round": None})
        assert fl_round_accuracy._value.get() == pytest.approx(0.0)
        assert fl_round_loss._value.get() == pytest.approx(0.0)


class TestPushMetrics:
    """push_metrics deve ser no-op quando FL_PUSHGATEWAY_URL não está definido."""

    def test_push_skips_without_env(self, monkeypatch):
        monkeypatch.delenv("FL_PUSHGATEWAY_URL", raising=False)
        from infrastructure.shared.metrics import push_metrics
        push_metrics(job="test")  # não deve levantar

    def test_push_calls_push_to_gateway_when_url_set(self, monkeypatch):
        monkeypatch.setenv("FL_PUSHGATEWAY_URL", "http://fake-gw:9091")
        with patch("prometheus_client.push_to_gateway") as mock_push:
            from infrastructure.shared.metrics import push_metrics
            push_metrics(job="test-job")
            mock_push.assert_called_once()
            call_kwargs = mock_push.call_args
            assert call_kwargs.args[0] == "http://fake-gw:9091"
            assert call_kwargs.kwargs["job"] == "test-job"

    def test_push_gateway_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("FL_PUSHGATEWAY_URL", "http://fake-gw:9091")
        with patch("prometheus_client.push_to_gateway", side_effect=OSError("connection refused")):
            from infrastructure.shared.metrics import push_metrics
            push_metrics(job="test-job")  # deve engolir o erro com warning
