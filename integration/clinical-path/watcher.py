"""
watcher.py
Monitoramento de diretório para ingestão automática de exames.

Observa um diretório por novos arquivos JSON no formato do endpoint
POST /api/exams/ingest e dispara um callback para cada arquivo novo.

Formato esperado:
    {
        "patient_id": "P001",
        "sex": "F",
        "age": 55.0,
        "exams": [
            {
                "exam_name": "WBC",
                "date": "2020-03-01",
                "value": 8.5,
                "phase": "IN",
                "ref_low": 4.0,
                "ref_high": 11.0
            }
        ]
    }

Uso standalone:
    from watcher import ExamFileWatcher
    w = ExamFileWatcher(Path("data/incoming"), on_new_file)
    w.start()   # bloqueante — rode em thread separada
    w.stop()
"""
import logging
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _NewJsonHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[Path], None]):
        super().__init__()
        self._callback = callback

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".json":
            return
        logger.info("new_exam_file_detected", extra={"path": str(path)})
        try:
            self._callback(path)
        except Exception as exc:
            logger.error(
                "watcher_callback_error",
                extra={"path": str(path), "error": str(exc)},
            )


class ExamFileWatcher:
    """Monitora um diretório e chama *callback* para cada novo arquivo .json."""

    def __init__(self, watch_dir: Path, callback: Callable[[Path], None]):
        self._watch_dir = Path(watch_dir)
        self._callback = callback
        self._observer = Observer()
        self._running = False

    def start(self) -> None:
        """Inicia o observer (bloqueante — rode em thread separada)."""
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        handler = _NewJsonHandler(self._callback)
        self._observer.schedule(handler, str(self._watch_dir), recursive=False)
        self._observer.start()
        self._running = True
        logger.info("watcher_started", extra={"watch_dir": str(self._watch_dir)})
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._observer.stop()
        self._observer.join()
        logger.info("watcher_stopped", extra={"watch_dir": str(self._watch_dir)})
