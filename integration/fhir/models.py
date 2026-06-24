"""
models.py — Contrato de entrada do FHIRExporter.

InferenceOutput carrega apenas o resultado do cálculo — sem dados clínicos brutos,
sem identidade de paciente. O correlation_token é gerado pelo sistema do hospital
chamador e ecoado de volta; o MOSAIC-FL não armazena o mapeamento token → paciente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid


@dataclass
class InferenceOutput:
    """Resultado de uma inferência do modelo federado.

    Não contém dados clínicos brutos nem identificador real de paciente.
    As probabilidades são uma propriedade do quadro clínico apresentado,
    não do indivíduo — o modelo aprende P(desfecho | quadro clínico) e não
    retém memória de nenhum paciente específico após o treinamento.

    Args:
        predictions:       Lista de (label_desfecho, probabilidade). Deve somar ~1.0.
        model_round:       Round FL em que o modelo foi treinado.
        temperature:       Temperatura de calibração (1.0 = sem calibração).
        ece:               Expected Calibration Error pós-calibração (< 0.05 = bom).
        correlation_token: Token opaco gerado pelo hospital chamador para correlacionar
                           a resposta com seu registro interno. Ecoado de volta no
                           RiskAssessment como subject.identifier. Se não fornecido,
                           um UUID é gerado automaticamente (o hospital não conseguirá
                           correlacionar — recomenda-se sempre passar o token).
        predicted_at:      Momento da inferência. Default: now(UTC).
    """

    predictions: List[Tuple[str, float]]
    model_round: int
    temperature: float
    ece: float
    correlation_token: str = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("predictions não pode ser vazio")
        if self.temperature <= 0:
            raise ValueError(f"temperature deve ser > 0, recebido: {self.temperature}")
        total = sum(p for _, p in self.predictions)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"probabilidades devem somar 1.0 (±0.01), soma={total:.4f}"
            )
