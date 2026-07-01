"""
inference_engine — Wrapper do modelo MOSAIC-FL para inferência single-patient em tempo real.

Tokenização alinhada ao treinamento:
  1. Resolve nome canônico do analito via knowledge.term_dictionary
  2. Classifica o valor via knowledge.analyte_references (HIGH/NORMAL/LOW/NO_REF)
  3. Compõe o token com o mesmo TokenMode usado no treinamento
  4. Mapeia para ID via vocabulário gravado junto com o checkpoint

Submódulos:
  compat.py          — disponibilidade de mosaicfl + fallback local de _make_token
  tokenization.py       — resolução de termos, classificação, records_to_tokens
  engine.py                — InferenceEngine
"""
from .compat import MAX_SEQ_LEN, VOCAB_SIZE
from .engine import InferenceEngine
from .tokenization import records_to_tokens

__all__ = ["InferenceEngine", "records_to_tokens", "VOCAB_SIZE", "MAX_SEQ_LEN"]
