"""
health_server.py — servidor HTTP mínimo de healthcheck para os daemons MOSAIC-FL.

Expõe GET /healthz → 200 ou 503 com JSON de status.
Roda em thread daemon; não bloqueia o processo principal nem impede o exit.

Uso:
    from infrastructure.health_server import HealthServer

    hs = HealthServer(port=8081)
    hs.start()          # inicia thread daemon
    hs.set_status("running", round=3, clients=5)
    hs.stop()           # opcional — o daemon morre com o processo
"""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

_UNHEALTHY = frozenset({"error", "stopped"})


class HealthServer:
    """Servidor HTTP mínimo que expõe /healthz e /metrics/round/{n} para os daemons."""

    def __init__(self, port: int = 8081) -> None:
        self._port = port
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"status": "starting"}
        self._round_metrics: dict[int, dict] = {}
        self._server: HTTPServer | None = None

    def set_status(self, status: str, **extra: Any) -> None:
        with self._lock:
            self._state = {"status": status, **extra}

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def set_round_metrics(self, round_num: int, metrics: dict) -> None:
        """Armazena métricas de um round concluído (consultável via HTTP)."""
        with self._lock:
            self._round_metrics[round_num] = metrics
        logger.debug("round_metrics_stored", extra={"round": round_num})

    def get_round_metrics(self, round_num: int) -> dict | None:
        with self._lock:
            return self._round_metrics.get(round_num)

    def start(self) -> None:
        """Inicia o servidor HTTP em thread daemon. Silencia erros de porta ocupada."""
        try:
            self._server = HTTPServer(("0.0.0.0", self._port), self._make_handler())
            thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="healthz",
            )
            thread.start()
            logger.info("health_server_started", extra={"port": self._port})
        except OSError as exc:
            logger.warning("health_server_unavailable", extra={"port": self._port, "error": str(exc)})

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def _make_handler(self) -> type:
        health_server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/livez":
                    self._send(200, {"status": "alive"})
                elif self.path in ("/readyz", "/healthz"):
                    state = health_server.get_state()
                    code = 503 if state.get("status") in _UNHEALTHY else 200
                    self._send(code, state)
                elif self.path.startswith("/metrics/round/"):
                    # GET /metrics/round/{n} — consultado pelo RoundDispatcher
                    try:
                        round_num = int(self.path.split("/")[-1])
                        metrics = health_server.get_round_metrics(round_num)
                        if metrics is not None:
                            self._send(200, metrics)
                        else:
                            self._send(404, {"available": False, "round": round_num})
                    except ValueError:
                        self._send(400, {"error": "round must be an integer"})
                elif self.path == "/metrics/rounds":
                    # GET /metrics/rounds — histórico completo (útil para debug)
                    with health_server._lock:
                        all_metrics = dict(health_server._round_metrics)
                    self._send(200, all_metrics)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send(self, code: int, data: dict) -> None:
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # silencia access log do HTTP server

        return _Handler
