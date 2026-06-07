"""
runner.py
Entrypoint do daemon mosaicfl_api — sobe o servidor FastAPI via uvicorn.

Uso:
    python -m infrastructure.mosaicfl_api
    python -m infrastructure.mosaicfl_api --port 9000
    FL_WATCH_DIR=data/incoming python -m infrastructure.mosaicfl_api

Variáveis de ambiente:
    FL_API_HOST            host (padrão: 0.0.0.0)
    FL_API_PORT            porta (padrão: 8080)
    FL_API_KEY             chave de autenticação (ausente = sem auth, só dev)
    FL_CHECKPOINT_DIR      diretório de checkpoints FL (padrão: checkpoints/)
    FL_CLINICALPATH_OUTPUT diretório de saída ClinicalPath (padrão: data/clinicalpath_output)
    FL_WATCH_DIR           diretório monitorado para novos exames JSON (opcional)
    FL_CORS_ORIGINS        origens CORS separadas por vírgula (padrão: *)
    FL_LOG_FORMAT          json | text (padrão: json)
    FL_LOG_LEVEL           DEBUG | INFO | WARNING (padrão: INFO)
"""
import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.shared.logging_setup import setup_logging as _setup_logging

logger = logging.getLogger(__name__)

_CP_DIR = Path(__file__).parent.parent.parent / "integration" / "clinical-path"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOSAIC-FL API daemon")
    p.add_argument("--host", default=os.getenv("FL_API_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("FL_API_PORT", "8080")))
    p.add_argument("--reload", action="store_true", help="hot-reload (desenvolvimento)")
    p.add_argument(
        "--watch-dir",
        default=os.getenv("FL_WATCH_DIR", ""),
        help="diretório para monitorar novos arquivos JSON de exames",
    )
    return p.parse_args()


def _start_watcher(watch_dir: Path, api_port: int) -> "ExamFileWatcher":  # type: ignore[name-defined]  # noqa: F821
    """Inicia o watcher em background. Retorna a instância para poder parar depois."""
    if str(_CP_DIR) not in sys.path:
        sys.path.insert(0, str(_CP_DIR))

    from watcher import ExamFileWatcher  # type: ignore[import]

    try:
        import httpx
    except ImportError:
        logger.warning("watcher_httpx_missing", extra={"action": "pip install httpx"})
        return None  # type: ignore[return-value]

    processed_dir = watch_dir.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    def _on_file(path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            httpx.post(
                f"http://localhost:{api_port}/api/exams/ingest",
                json=data,
                timeout=30,
            )
            path.rename(processed_dir / path.name)
            logger.info("watcher_file_ingested", extra={"file": path.name})
        except Exception as exc:
            logger.error("watcher_ingest_error", extra={"file": str(path), "error": str(exc)})

    watcher = ExamFileWatcher(watch_dir, _on_file)
    t = threading.Thread(target=watcher.start, daemon=True, name="exam-watcher")
    t.start()
    logger.info("file_watcher_started", extra={"watch_dir": str(watch_dir)})
    return watcher


def main() -> None:
    _setup_logging(log_file="api_daemon.log")
    args = _parse_args()

    logger.info(
        "api_startup",
        extra={
            "host": args.host,
            "port": args.port,
            "watch_dir": args.watch_dir or None,
            "checkpoint_dir": os.getenv("FL_CHECKPOINT_DIR", "checkpoints"),
            "auth_enabled": bool(os.getenv("FL_API_KEY")),
        },
    )

    watcher = None
    if args.watch_dir:
        watcher = _start_watcher(Path(args.watch_dir), args.port)

    try:
        import uvicorn
        uvicorn.run(
            "infrastructure.mosaicfl_api.service:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_config=None,  # usa nosso logging estruturado
        )
    except KeyboardInterrupt:
        logger.info("api_stopped")
    finally:
        if watcher is not None:
            watcher.stop()


if __name__ == "__main__":
    main()
