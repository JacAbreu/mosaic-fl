"""
mapper.py — Converte InferenceOutput em recurso FHIR R4 RiskAssessment.

Isolamento intencional:
  - Não importa nada de infrastructure/ (sem acesso ao banco)
  - Não importa PatientExport nem nenhum dado clínico bruto
  - Não recebe connection string nem dependências externas no construtor
  - A única entrada é InferenceOutput — arquiteturalmente impossível vazar
    dados clínicos por este módulo

Referência: https://www.hl7.org/fhir/riskassessment.html (R4)
"""
from __future__ import annotations

import uuid
from datetime import timezone
from typing import Any, Dict

from .models import InferenceOutput

_FHIR_VERSION = "4.0.1"
_PROFILE = "http://hl7.org/fhir/StructureDefinition/RiskAssessment"
_CORRELATION_SYSTEM = "urn:mosaicfl:correlation"
_METHOD_SYSTEM = "urn:mosaicfl:method"
_OUTCOME_SYSTEM = "urn:mosaicfl:outcome"


class FHIRExporter:
    """Produz recursos FHIR R4 a partir de InferenceOutput.

    Não tem estado — pode ser instanciado uma vez e reutilizado.
    Não acessa banco de dados nem sistema de arquivos.
    """

    def to_risk_assessment(self, output: InferenceOutput) -> Dict[str, Any]:
        """Converte InferenceOutput em dict compatível com FHIR R4 RiskAssessment.

        O dict retornado é serializável como JSON e válido contra o perfil
        http://hl7.org/fhir/StructureDefinition/RiskAssessment (R4).

        O campo subject.identifier contém apenas o correlation_token —
        um token efêmero gerado pelo hospital chamador. O MOSAIC-FL não
        armazena o mapeamento token → paciente.
        """
        predicted_at_iso = (
            output.predicted_at.astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        return {
            "resourceType": "RiskAssessment",
            "id": f"mosaicfl-{uuid.uuid4()}",
            "meta": {
                "profile": [_PROFILE],
            },
            "status": "final",
            "subject": {
                "identifier": {
                    "system": _CORRELATION_SYSTEM,
                    "value": output.correlation_token,
                }
            },
            "occurrenceDateTime": predicted_at_iso,
            "method": {
                "coding": [
                    {
                        "system": _METHOD_SYSTEM,
                        "code": "FedProx-BEHRT-v2",
                        "display": (
                            f"Federated FedProx + SimplifiedBEHRT "
                            f"— round {output.model_round}, T={output.temperature:.4f}"
                        ),
                    }
                ]
            },
            "prediction": [
                {
                    "outcome": {
                        "coding": [
                            {
                                "system": _OUTCOME_SYSTEM,
                                "code": self._slug(label),
                                "display": label,
                            }
                        ],
                        "text": label,
                    },
                    "probabilityDecimal": round(probability, 6),
                }
                for label, probability in output.predictions
            ],
            "note": [
                {
                    "text": (
                        f"ECE={output.ece:.4f} | "
                        f"T={output.temperature:.4f} | "
                        f"round={output.model_round} | "
                        f"fhir={_FHIR_VERSION}"
                    )
                }
            ],
        }

    @staticmethod
    def _slug(label: str) -> str:
        """Converte label de desfecho em código de sistema (ASCII, sem espaços)."""
        return (
            label.lower()
            .strip()
            .replace(" ", "_")
            .replace("ç", "c")
            .replace("ã", "a")
            .replace("â", "a")
            .replace("á", "a")
            .replace("à", "a")
            .replace("é", "e")
            .replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ô", "o")
            .replace("ú", "u")
            .replace("õ", "o")
        )
