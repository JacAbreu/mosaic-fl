"""
integration/fhir — Exportação FHIR R4 do MOSAIC-FL.

Expõe probabilidades de desfecho como RiskAssessment FHIR R4.
Não importa nada de infrastructure/ — sem acesso ao banco de dados.
"""
from .models import InferenceOutput
from .mapper import FHIRExporter

__all__ = ["InferenceOutput", "FHIRExporter"]
