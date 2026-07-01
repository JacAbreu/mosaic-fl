"""
exams_extract — Loads an exams CSV stream into metrics.exam_records.

Canonical normalization happens at ingestion:
  1. Raw analyte name is resolved to its canonical form via knowledge.term_dictionary.
  2. The value is classified (HIGH/NORMAL/LOW/NO_REF) using knowledge.analyte_references.
  3. canonical_ref_low, canonical_ref_high, and classification are stored in exam_records.

Two-phase workflow (run in order):
  1. scan_analytes() — streams the CSV, collects all analyte names, validates against
     term_dictionary. Registers unknowns as inactive. Does NOT insert data.
     Run this first and let the operator review pending terms before loading.
  2. load_exams() — streams the CSV chunk by chunk, resolves canonical names,
     classifies values, inserts records. Does not validate or block.

Expected source columns (resolved semantically, case-insensitive):
  patient_id, collection_date, analyte, result_text
  Optional: attendance_id, origin, exam_group, result_num, unit, reference_range

Submódulos:
  column_mapping.py  — resolução semântica de colunas (compartilhada scan/load)
  lookups.py           — cache de referências canônicas, resolução, classificação
  scan.py                — scan_analytes (fase 1 — validação sem carga)
  bulk_load.py              — load_exams + _process_chunk (fase 2 — carga)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .bulk_load import load_exams
from .scan import scan_analytes

__all__ = ["scan_analytes", "load_exams"]
