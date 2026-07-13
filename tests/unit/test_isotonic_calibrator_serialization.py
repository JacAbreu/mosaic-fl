"""
Testes para a serialização/reconstrução do IsotonicCalibrator — necessária para
persistir o calibrador no checkpoint (torch.save) e recarregá-lo na InferenceEngine
sem precisar refazer fit() (que exige modelo + calib_loader, indisponíveis na API).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.calibration import IsotonicCalibrator, TemperatureScaler


def _fitted_calibrator(num_classes: int = 3) -> IsotonicCalibrator:
    """Cria um IsotonicCalibrator ajustado sem precisar de model/loader — usa
    IsotonicRegression.fit() diretamente com dados sintéticos simples."""
    from sklearn.isotonic import IsotonicRegression

    calibrators = []
    for _ in range(num_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit([0.0, 0.5, 1.0], [0.0, 0.4, 1.0])
        calibrators.append(ir)
    return IsotonicCalibrator.from_calibrators(calibrators, num_classes)


class TestCalibratorsProperty:
    def test_empty_before_fit(self):
        iso = IsotonicCalibrator()
        assert iso.calibrators == []

    def test_returns_fitted_list(self):
        iso = _fitted_calibrator(num_classes=3)
        assert len(iso.calibrators) == 3


class TestFromCalibrators:
    def test_reconstructs_without_refit(self):
        original = _fitted_calibrator(num_classes=3)
        reconstructed = IsotonicCalibrator.from_calibrators(original.calibrators, num_classes=3)
        assert reconstructed._fitted is True
        assert reconstructed._num_classes == 3

    def test_empty_calibrators_marks_unfitted(self):
        iso = IsotonicCalibrator.from_calibrators([], num_classes=0)
        assert iso._fitted is False

    def test_calibrate_probs_works_after_reconstruction(self):
        original = _fitted_calibrator(num_classes=3)
        reconstructed = IsotonicCalibrator.from_calibrators(original.calibrators, num_classes=3)

        probs = torch.tensor([[0.5, 0.3, 0.2]])
        result = reconstructed.calibrate_probs(probs)

        assert result.shape == (1, 3)
        # renormalizado para simplex válido
        assert abs(float(result.sum()) - 1.0) < 1e-4


class TestCalibrateProbsVsCalibrate:
    def test_calibrate_probs_matches_calibrate_on_logits(self):
        """calibrate(logits) faz softmax internamente; calibrate_probs(probs) não —
        aplicados ao mesmo par logits/softmax(logits), devem produzir o mesmo resultado."""
        iso = _fitted_calibrator(num_classes=3)
        logits = torch.tensor([[2.0, 1.0, 0.1]])
        probs = torch.softmax(logits, dim=1)

        via_logits = iso.calibrate(logits)
        via_probs  = iso.calibrate_probs(probs)

        assert torch.allclose(via_logits, via_probs, atol=1e-6)

    def test_calibrate_probs_raises_when_not_fitted(self):
        iso = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            iso.calibrate_probs(torch.tensor([[0.5, 0.3, 0.2]]))


class TestFitFromLogits:
    """Ajuste a partir de logits/labels pré-calculados — usado por
    FedProxClient._fit_local_calibrator() para evitar um segundo forward pass."""

    def test_isotonic_fit_from_logits(self):
        logits = torch.randn(30, 3)
        labels = torch.randint(0, 3, (30,))
        iso = IsotonicCalibrator().fit_from_logits(logits, labels, num_classes=3)
        assert iso._fitted is True
        assert len(iso.calibrators) == 3

    def test_temperature_fit_from_logits(self):
        logits = torch.randn(30, 3)
        labels = torch.randint(0, 3, (30,))
        scaler = TemperatureScaler().fit_from_logits(logits, labels)
        assert scaler.T > 0

    def test_fit_and_fit_from_logits_agree(self):
        """fit(model, loader) e fit_from_logits(logits, labels) devem convergir pro
        mesmo T quando alimentados com os mesmos dados — confirma que a refatoração
        não mudou o comportamento numérico."""
        from torch.utils.data import DataLoader, TensorDataset
        from mosaicfl.core.model import SimplifiedBEHRT
        from mosaicfl.core.config import MODEL_CFG

        model = SimplifiedBEHRT(use_cls_token=True)
        model.eval()
        x = torch.randint(1, MODEL_CFG.vocab_size, (16, 8))
        y = torch.randint(0, MODEL_CFG.num_classes, (16,))
        dia = torch.randint(0, 100, (16, 8))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=16)

        scaler_via_fit = TemperatureScaler().fit(model, loader, device="cpu")

        with torch.no_grad():
            logits = model(x, dia_relativo=dia)
        scaler_via_logits = TemperatureScaler().fit_from_logits(logits, y, device="cpu")

        assert abs(scaler_via_fit.T - scaler_via_logits.T) < 1e-4


class TestThresholdExportImport:
    """Serialização usada para federar a calibração isotônica: cada cliente exporta
    os breakpoints pós-PAV (estatística comprimida, não dado bruto), o servidor
    agrega (aggregate_calibration em federated.py) e reconstrói via
    from_pooled_thresholds()."""

    def test_export_thresholds_shape(self):
        iso = _fitted_calibrator(num_classes=3)
        thresholds = iso.export_thresholds()
        assert len(thresholds) == 3
        for x_list, y_list in thresholds:
            assert isinstance(x_list, list)
            assert isinstance(y_list, list)
            assert len(x_list) == len(y_list)

    def test_from_pooled_thresholds_reconstructs_usable_calibrator(self):
        pooled = [
            ([0.1, 0.5, 0.9], [0.0, 0.5, 1.0]),
            ([0.2, 0.6], [0.0, 1.0]),
        ]
        iso = IsotonicCalibrator.from_pooled_thresholds(pooled, num_classes=2)
        assert iso._fitted is True
        result = iso.calibrate_probs(torch.tensor([[0.5, 0.5]]))
        assert result.shape == (1, 2)
        assert abs(float(result.sum()) - 1.0) < 1e-4

    def test_export_then_pool_then_reconstruct_roundtrip(self):
        """Simula o fluxo completo: cliente exporta -> servidor concatena (2 clientes,
        mesma classe) -> servidor reconstrói. Não deve levantar exceção nem produzir
        um calibrador vazio."""
        iso_a = _fitted_calibrator(num_classes=2)
        iso_b = _fitted_calibrator(num_classes=2)

        thresholds_a = iso_a.export_thresholds()
        thresholds_b = iso_b.export_thresholds()

        pooled = []
        for (xa, ya), (xb, yb) in zip(thresholds_a, thresholds_b):
            pooled.append((xa + xb, ya + yb))

        merged = IsotonicCalibrator.from_pooled_thresholds(pooled, num_classes=2)
        assert merged._fitted is True
        assert len(merged.calibrators) == 2
