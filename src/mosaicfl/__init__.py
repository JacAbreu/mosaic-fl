"""
Mosaic-FL / MOSAICO-FL
Módulo de Predição Federada para Trajetórias Clínicas
Extensão preditiva do ClinicalPath (Linhares et al., 2023)
"""

__version__ = "0.1.0"
__author__ = "Jacqueline Abreu do N. T. R. Lopes"
__institution__ = "ICMC/USP - São Carlos"

# Camada 1: Dados e Pré-processamento (Experimento 1)
from .preprocess import EHRPreprocessor, split_by_institution

# Camada 2: Modelo BEHRT Simplificado
from .model import SimplifiedBEHRT

# Camada 3: Aprendizado Federado (Flower + FedProx)
from .client import FedProxClient
from .server import start_server, ConvergenceTracker

# Camada 4: RAG para Justificativa Diagnóstica
from .rag_system import ClinicalRAG

# Camada 5: Extração de Padrões do BEHRT para o RAG
from .extract_patterns import BEHRTPatternExtractor

# Configurações globais
from .config import *

__all__ = [
    "EHRPreprocessor",
    "split_by_institution",
    "SimplifiedBEHRT",
    "FedProxClient",
    "start_server",
    "ConvergenceTracker",
    "ClinicalRAG",
    "BEHRTPatternExtractor",
    "__version__",
]
