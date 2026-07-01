"""column_mapping.py — Resolução semântica das colunas do CSV de exames (compartilhada entre scan e load)."""
from integration.column_resolver import CLINICAL_SEMANTIC_MAP, ColumnResolver

_REQUIRED = {"patient_id", "collection_date", "analyte", "result_text"}
_RESOLVER = ColumnResolver(CLINICAL_SEMANTIC_MAP, required=_REQUIRED)

_BATCH_SIZE = 500
