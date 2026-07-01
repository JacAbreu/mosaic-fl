"""
serialization.py — Serialização de checkpoints (pesos + vocab + metadados) e DDL SQLite.

_model_version — fingerprint SHA-256 dos pesos (12 hex chars)
_serialize      — empacota state_dict + vocab + metadados em bytes (torch.save)
_deserialize    — desempacota bytes de volta em dict
"""
import io
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict

import torch

_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS fl_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    round       INTEGER NOT NULL,
    accuracy    REAL    NOT NULL DEFAULT 0.0,
    loss        REAL    NOT NULL DEFAULT 0.0,
    model_bytes BLOB    NOT NULL,
    sha256      TEXT    NOT NULL,
    vocab_size  INTEGER NOT NULL DEFAULT 0,
    training_id INTEGER,
    created_at  TEXT    NOT NULL
)
"""

_CREATE_SQLITE_TRAININGS = """
CREATE TABLE IF NOT EXISTS fl_trainings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm     TEXT NOT NULL DEFAULT 'FedAvg',
    log_file      TEXT NOT NULL DEFAULT '',
    n_rounds_max  INTEGER NOT NULL DEFAULT 120,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    n_rounds_done INTEGER,
    best_round    INTEGER,
    best_accuracy REAL,
    converged     INTEGER
)
"""


def _model_version(state_dict: OrderedDict) -> str:
    """Fingerprint SHA-256 dos pesos — 12 hex chars. Mesmo modelo → mesmo hash."""
    import hashlib
    h = hashlib.sha256()
    for v in state_dict.values():
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def _serialize(
    state_dict: OrderedDict,
    vocab: Dict[str, int],
    temperature: float = 1.0,
    checkpoint_round: int = 0,
) -> bytes:
    buf = io.BytesIO()
    torch.save(
        {
            "model_state":      state_dict,
            "vocab":            vocab,
            "temperature":      temperature,
            "checkpoint_round": checkpoint_round,
            "checkpoint_at":    datetime.now(timezone.utc).isoformat(),
            "model_version":    _model_version(state_dict),
        },
        buf,
    )
    return buf.getvalue()


def _deserialize(data: bytes) -> Dict:
    buf = io.BytesIO(data)
    # weights_only=False: checkpoint contém vocab (dict str→int), temperatura e metadados além de tensors
    return torch.load(buf, map_location="cpu", weights_only=False)
