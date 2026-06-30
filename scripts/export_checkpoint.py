"""
Exporta o melhor checkpoint do CheckpointStore (SQLite ou PostgreSQL)
para um arquivo .pt que a API pode carregar via FL_CHECKPOINT_DIR.

Uso:
    python scripts/export_checkpoint.py
    FL_TRAINING_ID=5 python scripts/export_checkpoint.py
    FL_DB_URL=postgresql://... python scripts/export_checkpoint.py

Saída: checkpoints/best_model.pt
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from infrastructure.shared.checkpoint_store import get_checkpoint_store

db_url      = os.getenv("FL_DB_URL", "")
training_id = int(os.getenv("FL_TRAINING_ID", "0")) or None
out_path    = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints")) / "best_model.pt"

store = get_checkpoint_store(db_url)
data  = store.load_best(training_id=training_id)

if data is None:
    print("ERRO: nenhum checkpoint encontrado no banco.", file=sys.stderr)
    print("Execute 'make training-full' antes de exportar.", file=sys.stderr)
    sys.exit(1)

out_path.parent.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
torch.save(data, buf)
out_path.write_bytes(buf.getvalue())

print(
    f"Checkpoint exportado: {out_path}\n"
    f"  round    = {data.get('checkpoint_round')}\n"
    f"  version  = {data.get('model_version')}\n"
    f"  vocab    = {len(data.get('vocab', {})):,} tokens\n"
    f"  temp     = {data.get('temperature', 1.0):.4f}"
)
