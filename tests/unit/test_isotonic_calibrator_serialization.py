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

from mosaicfl.core.calibration import IsotonicCalibrator


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
