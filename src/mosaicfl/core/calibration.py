"""
calibration.py — Calibração probabilística pós-treinamento.

Dois métodos disponíveis:

  TemperatureScaler — temperature scaling (Guo et al., ICML 2017).
    Um único escalar T > 0 divide os logits antes do softmax.
    Simples, mas falha quando o viés de calibração não é monotônico uniforme.
    8 experimentos neste projeto confirmam que T>1 piora ECE (padrão subconfiante não-uniforme).

  IsotonicCalibrator — calibração isotônica OvR (Zadrozny & Elkan, 2002).
    Aprende uma função monotônica não-paramétrica por classe via pool adjacent violators (PAV).
    Cada bin é ajustado independentemente — captura padrões não-uniformes que temperature
    scaling não resolve. Recomendado quando ECE piora após temperature scaling.
"""
import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Parametrização em espaço log: T = exp(log_T), garantindo T > 0 matematicamente.
    Isso evita que o LBFGS (método de segunda ordem) salte para valores negativos —
    problema que ocorre quando o clamp(min=ε) zera o gradiente e o otimizador
    interpreta erroneamente que convergiu fora do domínio válido.
    """

    def __init__(self) -> None:
        super().__init__()
        # log(1.5) ≈ 0.405 — ponto inicial equivalente ao anterior
        self.log_temperature = nn.Parameter(torch.tensor([0.405]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(
        self,
        model: nn.Module,
        calib_loader: "torch.utils.data.DataLoader",
        device: str = "cpu",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """Otimiza T minimizando NLL no calib_loader via LBFGS.

        O modelo não é modificado — apenas self.log_temperature é atualizado.
        Recomenda-se um conjunto de calibração holdout separado do treino e do teste.
        Em simulação acadêmica, o test_loader pode ser reutilizado com ressalva documentada.
        """
        model.eval()
        model.to(device)
        self.to(device)

        logits_list: list = []
        labels_list: list = []
        with torch.no_grad():
            for batch_x, batch_y, batch_dia in calib_loader:
                batch_x = batch_x.to(device)
                batch_dia = batch_dia.to(device)
                logits_list.append(model(batch_x, dia_relativo=batch_dia).cpu())
                labels_list.append(batch_y.cpu())

        if not logits_list:
            logger.warning("temperature_scaling_fit: calib_loader vazio — T mantido em 1.0")
            return self

        all_logits = torch.cat(logits_list).to(device)
        all_labels = torch.cat(labels_list).to(device)

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = nll(all_logits / self.log_temperature.exp(), all_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        logger.info("temperature_scaling_fit T=%.4f", self.T)
        return self

    @property
    def T(self) -> float:
        """Valor escalar de temperatura após calibração (sempre positivo)."""
        return float(self.log_temperature.exp().item())


class IsotonicCalibrator:
    """
    Calibração isotônica OvR (One-vs-Rest) para classificação multiclasse.

    Aprende uma função monotônica não-paramétrica por classe via pool adjacent violators (PAV),
    mapeando confiança softmax bruta → probabilidade calibrada por classe.

    Vantagem sobre temperature scaling: cada bin é ajustado independentemente,
    sem assumir que o viés é uniforme em toda a curva de confiabilidade.

    Limitação: requer conjunto de calibração suficientemente grande (≥ 50 amostras por classe).
    Com N < 1000 o ajuste pode ser instável; documentar como limitação metodológica.

    Referência: Zadrozny & Elkan (2002) — "Transforming classifier scores into accurate
    multiclass probability estimates."
    """

    def __init__(self) -> None:
        self._calibrators: List = []
        self._num_classes: int = 0
        self._fitted: bool = False

    def fit(
        self,
        model: nn.Module,
        calib_loader: "torch.utils.data.DataLoader",
        device: str = "cpu",
        num_classes: int = 5,
    ) -> "IsotonicCalibrator":
        """Ajusta um IsotonicRegression por classe no calib_loader."""
        from sklearn.isotonic import IsotonicRegression

        model.eval()
        model.to(device)
        self._num_classes = num_classes

        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in calib_loader:
                bx, by, bdia = batch[0].to(device), batch[1], batch[2].to(device)
                logits_list.append(model(bx, dia_relativo=bdia).cpu())
                labels_list.append(by.cpu())

        if not logits_list:
            logger.warning("isotonic_calibration_fit: calib_loader vazio — calibrador não ajustado")
            return self

        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list).numpy()
        probs      = F.softmax(all_logits, dim=1).numpy()

        self._calibrators = []
        for c in range(num_classes):
            binary_labels = (all_labels == c).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, c], binary_labels)
            self._calibrators.append(ir)

        self._fitted = True
        logger.info("isotonic_calibration_fit num_classes=%d n_samples=%d", num_classes, len(all_labels))
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aplica calibração isotônica e renormaliza para simplex válido.

        Returns:
            Tensor (N, num_classes) com probabilidades calibradas somando ~1.
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator não foi ajustado. Chame fit() primeiro.")

        probs     = F.softmax(logits, dim=1).detach().cpu().numpy()
        cal_probs = np.stack(
            [self._calibrators[c].predict(probs[:, c]) for c in range(self._num_classes)],
            axis=1,
        ).clip(0.0, 1.0)

        # renormalizar — isotônica OvR não garante que a soma das classes = 1
        row_sums  = cal_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        cal_probs = cal_probs / row_sums

        return torch.from_numpy(cal_probs.astype(np.float32))

    def compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
        """ECE (Expected Calibration Error) com probabilidades calibradas isotonicamente."""
        cal_probs = self.calibrate(logits)
        confidences, predictions = cal_probs.max(dim=1)
        accuracies  = predictions.eq(labels)
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_acc  = accuracies[mask].float().mean().item()
                bin_conf = confidences[mask].mean().item()
                ece     += (mask.float().mean().item()) * abs(bin_acc - bin_conf)
        return round(ece, 4)
