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
    """Servidor HTTP mínimo que expõe /healthz para probes de Kubernetes e Docker."""

    def __init__(self, port: int = 8081) -> None:
        self._port = port
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"status": "starting"}
        self._server: HTTPServer | None = None

    def set_status(self, status: str, **extra: Any) -> None:
        with self._lock:
            self._state = {"status": status, **extra}

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

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
                    # Liveness: processo está vivo se consegue responder.
                    # Nunca retorna 503 — se o processo morreu, a probe falha
                    # por timeout, não por código de status.
                    self._send(200, {"status": "alive"})
                elif self.path in ("/readyz", "/healthz"):
                    # Readiness: reflete estado funcional do daemon.
                    state = health_server.get_state()
                    code = 503 if state.get("status") in _UNHEALTHY else 200
                    self._send(code, state)
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
