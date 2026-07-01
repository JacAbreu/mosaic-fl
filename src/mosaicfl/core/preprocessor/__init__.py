"""
mosaicfl.core.preprocessor — Pré-processamento e pipelines de sequência para o BEHRT.

Submódulos:
  tokens.py             — TokenMode, _make_token (composição de vocabulário)
  outcomes.py            — _map_outcome (mapeamento clínico de desfecho)
  legacy_csv.py           — EHRPreprocessor, split_by_institution (caminho CSV/sintético)
  sequence_pipeline.py    — SequencePipeline (pipeline de produção via banco)
  legacy_reference.py     — SequencePipelineInicial (referência histórica, não usado em produção)
"""
from .legacy_csv import EHRPreprocessor, split_by_institution
from .legacy_reference import SequencePipelineInicial
from .outcomes import _map_outcome
from .sequence_pipeline import _SQL_ATENDIMENTOS, SequencePipeline
from .tokens import TokenMode, _make_token

__all__ = [
    "EHRPreprocessor",
    "split_by_institution",
    "TokenMode",
    "_make_token",
    "_map_outcome",
    "SequencePipeline",
    "SequencePipelineInicial",
]
