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
_CHECKPOINT_DIR    = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
_OUTPUT_DIR        = Path(os.getenv("FL_CLINICALPATH_OUTPUT", "data/clinicalpath_output"))

# FL_CHECKPOINT_SOURCE controla a ordem de carregamento do modelo na inicialização:
#   "db"   (padrão) — banco primário, arquivos como fallback
#   "file"          — arquivos primários, banco como fallback
# Trocar para "file" é útil em deploy offline (sem acesso ao banco de treinamento).
_CHECKPOINT_SOURCE = os.getenv("FL_CHECKPOINT_SOURCE", "db")

# FL_TRAINING_ID (opcional) — carrega o checkpoint de um training_id específico.
# Se não definido, tenta ler experiments/last_federated_training_id.txt gravado
# automaticamente pelo make training-full (fase 3/4, federado BPSP+HSL).
# Sem nenhuma das duas fontes, load_best() usa o checkpoint com maior accuracy
# global, que pode pertencer a qualquer fase (BPSP-only, HSL-only, etc.).
_FEDERATED_ID_FILE = Path("experiments/last_federated_training_id.txt")

def _resolve_inference_training_id() -> Optional[int]:
    raw = os.getenv("FL_TRAINING_ID")
    if raw:
        return int(raw)
    if _FEDERATED_ID_FILE.exists():
        try:
            val = _FEDERATED_ID_FILE.read_text(encoding="utf-8").strip()
            if val.isdigit():
                return int(val)
        except Exception:
            pass
    return None

_INFERENCE_TRAINING_ID: Optional[int] = _resolve_inference_training_id()

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
_rag = None  # type: Optional["mosaicfl.core.rag.ClinicalRAG"]  — instanciado sob demanda


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_checkpoint() -> Optional[Path]:
    """Retorna o .pt mais recente em FL_CHECKPOINT_DIR, ou None se não houver."""
    if not _CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(_CHECKPOINT_DIR.glob("round_*.pt"))
    if ckpts:
        return ckpts[-1]
    # make export-checkpoint grava best_model.pt
    best = _CHECKPOINT_DIR / "best_model.pt"
    return best if best.exists() else None


def _verify_checkpoint_integrity(path: Path) -> bool:
    """Verifica SHA-256 do checkpoint contra arquivo .sha256 (se existir)."""
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists():
        return True
    actual   = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = hash_path.read_text(encoding="utf-8").strip()
    return actual == expected


def _load_from_store(engine: "InferenceEngine", db_url: str, training_id: Optional[int] = None) -> bool:
    """Carrega pesos do CheckpointStore (SQLite/PostgreSQL). Retorna True se OK.

    training_id — quando fornecido (via FL_TRAINING_ID), garante que a API serve
    o modelo de uma fase específica do pipeline (ex.: federado, não BPSP-only).
    Sem training_id, load_best() usa o checkpoint com maior accuracy global, que
    pode pertencer a qualquer fase do make training-full.
    """
    try:
        from infrastructure.shared.checkpoint_store import get_checkpoint_store
        store     = get_checkpoint_store(db_url)
        ckpt_data = store.load_best(training_id=training_id)
        if ckpt_data:
            engine.load_from_store(ckpt_data)
            logger.info(
                "startup_checkpoint_source=store vocab=%d training_id=%s",
                len(engine._vocab), training_id,
            )
            return True
        logger.warning(
            "startup_store_empty training_id=%s — nenhum checkpoint no banco; "
            "execute 'make training-full' antes de iniciar a API",
            training_id,
        )
        return False
    except Exception as exc:
        logger.warning("startup_store_unavailable: %s", exc)
        return False


def _load_from_file(engine: "InferenceEngine") -> bool:
    """Carrega pesos do arquivo .pt mais recente em FL_CHECKPOINT_DIR. Retorna True se OK."""
    ckpt_path = _latest_checkpoint()
    if ckpt_path is None:
        logger.warning(
            "startup_file_not_found — %s não contém round_*.pt nem best_model.pt; "
            "execute 'make export-checkpoint' para materializar o arquivo",
            _CHECKPOINT_DIR,
        )
        return False
    try:
        engine.reload(ckpt_path)
        logger.info("startup_checkpoint_source=file path=%s vocab=%d", ckpt_path, len(engine._vocab))
        return True
    except Exception as exc:
        logger.warning("startup_file_invalid path=%s: %s", ckpt_path, exc)
        return False


def _get_engine() -> "InferenceEngine":
    global _engine
    if _engine is None:
        db_url  = os.getenv("FL_DB_URL")
        # Inicializa sem pesos — carrega referências clínicas do banco independente da fonte do modelo
        _engine = InferenceEngine(checkpoint_path=None, db_url=db_url)

        if _CHECKPOINT_SOURCE == "db":
            primary   = lambda: _load_from_store(_engine, db_url, _INFERENCE_TRAINING_ID) if db_url else False
            secondary = lambda: _load_from_file(_engine)
        else:
            primary   = lambda: _load_from_file(_engine)
            secondary = lambda: _load_from_store(_engine, db_url, _INFERENCE_TRAINING_ID) if db_url else False

        loaded = primary() or secondary()

        if not loaded:
            logger.error(
                "startup_no_model — API iniciou SEM modelo carregado "
                "(FL_CHECKPOINT_SOURCE=%s). Predições retornarão zeros até recarregar.",
                _CHECKPOINT_SOURCE,
            )

    return _engine


def _get_rag():
    """Instancia ClinicalRAG sob demanda (1ª chamada) — carrega embedding model +
    backend LLM (Ollama/HuggingFace), caro de recriar a cada request. Chamador
    (routers/prediction.py) deve tratar exceções — RAG é enriquecimento opcional
    da predição, uma falha aqui não deve derrubar /api/predict."""
    global _rag
    if _rag is None:
        from mosaicfl.core.rag import ClinicalRAG
        _rag = ClinicalRAG(db_url=os.getenv("FL_DB_URL", ""))
    return _rag


def _patient_lock(pid: str):
    import asyncio
    if pid not in _patient_locks:
        _patient_locks[pid] = asyncio.Lock()
    return _patient_locks[pid]
