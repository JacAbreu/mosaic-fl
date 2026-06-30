"""
state.py — Singletons e estado mutable compartilhado entre os routers.

Separado de service.py para que cada router importe o módulo (não valores),
permitindo que testes substituam _engine e _db por mocks via:
    from infrastructure.mosaicfl_api import state
    state._engine = mock_engine
    state._db = PatientDB(test_path)
"""
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths de configuração (mutáveis para testes) ──────────────────────────────
_CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
_OUTPUT_DIR     = Path(os.getenv("FL_CLINICALPATH_OUTPUT", "data/clinicalpath_output"))

# ── Integration: ClinicalPath (módulo com hífen — não é package Python) ──────
_INTEGRATION_DIR = Path(__file__).parent.parent.parent / "integration" / "clinical-path"
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))

from exporter import ClinicalPathExporter                                   # noqa: E402
from models import (                                                         # noqa: E402
    ClinicalPhase, ExamRecord, PatientExport, ProbabilityEstimate, RiskPrediction,
)

# ── Integration: FHIR R4 ─────────────────────────────────────────────────────
_FHIR_DIR = Path(__file__).parent.parent.parent / "integration" / "fhir"
if str(_FHIR_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_FHIR_DIR.parent))

from integration.fhir import FHIRExporter, InferenceOutput                  # noqa: E402

# ── Local modules ─────────────────────────────────────────────────────────────
from .db import PatientDB                                                    # noqa: E402
from .inference_engine import InferenceEngine                                # noqa: E402

# ── Singletons ────────────────────────────────────────────────────────────────
_db:              PatientDB           = PatientDB()
_exporter:        ClinicalPathExporter = ClinicalPathExporter()
_fhir_exporter:   FHIRExporter        = FHIRExporter()
_engine:          Optional[InferenceEngine] = None
_patient_locks:   dict               = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_checkpoint() -> Optional[Path]:
    if not _CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(_CHECKPOINT_DIR.glob("round_*.pt"))
    return ckpts[-1] if ckpts else None


def _verify_checkpoint_integrity(path: Path) -> bool:
    """Verifica SHA-256 do checkpoint contra arquivo .sha256 (se existir)."""
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists():
        return True
    actual   = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = hash_path.read_text(encoding="utf-8").strip()
    return actual == expected


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        db_url    = os.getenv("FL_DB_URL")
        ckpt_path = _latest_checkpoint()

        _engine = InferenceEngine(checkpoint_path=ckpt_path, db_url=db_url)

        # Fallback: carrega do CheckpointStore quando não há arquivo .pt disponível.
        # O pipeline de treinamento persiste checkpoints no banco (SQLite/PostgreSQL);
        # a API usa esse caminho se FL_CHECKPOINT_DIR não tiver arquivos round_*.pt.
        if not _engine._vocab and db_url:
            try:
                from infrastructure.shared.checkpoint_store import get_checkpoint_store
                store     = get_checkpoint_store(db_url)
                ckpt_data = store.load_best()
                if ckpt_data:
                    _engine.load_from_store(ckpt_data)
                    logger.info("startup_checkpoint_loaded_from_store")
                else:
                    logger.warning(
                        "startup_no_checkpoint — execute 'make training-full' antes de iniciar a API"
                    )
            except Exception as exc:
                logger.warning("startup_checkpoint_store_unavailable: %s", exc)

    return _engine


def _patient_lock(pid: str):
    import asyncio
    if pid not in _patient_locks:
        _patient_locks[pid] = asyncio.Lock()
    return _patient_locks[pid]
