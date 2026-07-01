"""serialization.py — DDL das tabelas de métricas e desserialização de campos JSON."""
import json
from typing import Dict

_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS fl_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    round                   INTEGER NOT NULL,
    checkpoint_sha256       TEXT,
    accuracy                REAL,
    loss                    REAL,
    macro_auc               REAL,
    macro_f1                REAL,
    ece                     REAL,
    per_class_auc           TEXT,
    per_class_f1            TEXT,
    rag_precision_at_k      REAL,
    rag_k                   INTEGER,
    rag_per_class_precision TEXT,
    data_source             TEXT NOT NULL DEFAULT 'synthetic',
    evaluated_at            TEXT NOT NULL
)
"""

_CREATE_POSTGRES = """
CREATE TABLE IF NOT EXISTS metrics.fl_metrics (
    id                      SERIAL PRIMARY KEY,
    round                   INTEGER     NOT NULL,
    checkpoint_sha256       TEXT,
    accuracy                REAL,
    loss                    REAL,
    macro_auc               REAL,
    macro_f1                REAL,
    ece                     REAL,
    per_class_auc           JSONB,
    per_class_f1            JSONB,
    rag_precision_at_k      REAL,
    rag_k                   INTEGER,
    rag_per_class_precision JSONB,
    data_source             TEXT        NOT NULL DEFAULT 'synthetic',
    evaluated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _row_to_dict(row: Dict) -> Dict:
    """Desserializa campos JSON armazenados como string."""
    for field in ("per_class_auc", "per_class_f1", "rag_per_class_precision"):
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return row
