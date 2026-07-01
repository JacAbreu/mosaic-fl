"""checkpoint_io.py — Carga de vocabulário padrão e persistência de checkpoint com hash de integridade."""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict

import torch

from .config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def _load_standard_vocab() -> Dict:
    """Carrega standard_vocab.json como fallback quando não há checkpoint."""
    candidates = [
        os.getenv("FL_VOCAB_PATH"),
        str(CHECKPOINT_DIR / "standard_vocab.json"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                with open(path, encoding="utf-8") as f:
                    vocab = json.load(f)
                logger.info("standard_vocab_loaded path=%s size=%d", path, len(vocab))
                return vocab
            except Exception as exc:
                logger.warning("standard_vocab_load_error path=%s error=%s", path, exc)
    logger.warning(
        "no_standard_vocab — execute scripts/build_standard_vocab.py antes do treinamento; "
        "inferência será inoperante até o primeiro checkpoint com vocab"
    )
    return {}


def _save_checkpoint(path: Path, state: dict) -> None:
    """Salva checkpoint e grava SHA-256 para verificação de integridade."""
    torch.save(state, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(sha256, encoding="utf-8")
