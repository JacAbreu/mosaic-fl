"""compat.py — Disponibilidade do pacote mosaicfl e fallback local de _make_token.

Fallback é usado quando mosaicfl não está instalado no ambiente (ex: API standalone) —
espelha exatamente a lógica de mosaicfl.core.preprocessor._make_token.
"""
import os

_DEFAULT_MC_SAMPLES = int(os.getenv("FL_MC_SAMPLES", "50"))

_MOSAICFL_AVAILABLE = False
_VOCAB_SIZE = 10000
_MAX_SEQ_LEN = 128

try:
    from mosaicfl.core.config import MODEL_CFG
    from mosaicfl.core.model import SimplifiedBEHRT
    from mosaicfl.core.preprocessor import TokenMode, _make_token
    _MOSAICFL_AVAILABLE = True
    _VOCAB_SIZE = MODEL_CFG.vocab_size
    _MAX_SEQ_LEN = MODEL_CFG.max_seq_len
except Exception:
    # Fallback local — espelha exatamente a lógica do preprocessor.py
    def _make_token(analyte: str, classification: str, mode: str = "FULL") -> str:
        if mode == "ANALYTE_ONLY":
            return analyte
        if mode == "CLASS_ONLY":
            return classification
        if classification == "NO_REF":
            return analyte
        return f"{analyte}_{classification}"

VOCAB_SIZE  = _VOCAB_SIZE
MAX_SEQ_LEN = _MAX_SEQ_LEN
