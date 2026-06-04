"""
mosaicfl — MOSAIC-FL: Predição Clínica Federada com BEHRT + RAG.

Entrypoints (raiz do repositório):
  python run_experiments.py     → mosaicfl.experiments.runner → mosaicfl.v1.*
  python run_experiments_v2.py  → script na raiz → mosaicfl.v2.*

Pacotes:
  mosaicfl.v1  — experimentos sintéticos / validação TCC
  mosaicfl.v2  — dados reais (SGBD/CSV), modelo e FL corrigidos
"""

__version__ = "0.2.0"
