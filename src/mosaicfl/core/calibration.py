"""
calibration.py — Temperature scaling pós-treinamento.

Temperature scaling é o método mais simples e eficaz de calibração probabilística
para redes neurais (Guo et al., ICML 2017). Aprende um único escalar T > 0 que
divide os logits antes do softmax, minimizando a NLL no conjunto de calibração holdout.

T > 1 → softmax mais suave (modelo estava confiante demais)
T < 1 → softmax mais afiado (modelo estava sub-confiante)
T = 1 → sem calibração (identidade)
"""
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import torch.utils.data

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """
    Calibrador por temperature scaling.

    Uso típico:
        scaler = TemperatureScaler()
        scaler.fit(global_model, calib_loader, device="cpu")
        calibrated_logits = scaler(raw_logits)
        T = scaler.T  # persiste no checkpoint
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(
        self,
        model: nn.Module,
        calib_loader: "torch.utils.data.DataLoader",
        device: str = "cpu",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """Otimiza T minimizando NLL no calib_loader via LBFGS.

        O modelo não é modificado — apenas self.temperature é atualizado.
        Recomenda-se um conjunto de calibração holdout separado do treino e do teste.
        Em simulação acadêmica, o test_loader pode ser reutilizado com ressalva documentada.
        """
        model.eval()
        model.to(device)
        self.to(device)

        logits_list: list = []
        labels_list: list = []
        with torch.no_grad():
            for batch_x, batch_y in calib_loader:
                batch_x = batch_x.to(device)
                logits_list.append(model(batch_x).cpu())
                labels_list.append(batch_y.cpu())

        if not logits_list:
            logger.warning("temperature_scaling_fit: calib_loader vazio — T mantido em 1.0")
            return self

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = nll(all_logits / self.temperature.clamp(min=1e-3), all_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        logger.info("temperature_scaling_fit T=%.4f", self.T)
        return self

    @property
    def T(self) -> float:
        """Valor escalar de temperatura após calibração."""
        return float(self.temperature.item())
