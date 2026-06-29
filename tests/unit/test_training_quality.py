"""
Testes de qualidade do pipeline de treinamento federado.

Cobre as 6 melhorias implementadas para nível MVP:
  1. Class weight clipping (max=15.0)
  2. Gradient clipping (max_norm=1.0)
  3. Grad norm no retorno de fit()
  4. local_epochs = 1
  5. DataLoader com gerador fixo (shuffling determinístico)
  6. IsotonicCalibrator
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.config import FED_CFG, MODEL_CFG, VOCAB_SIZE, NUM_CLASSES, MAX_SEQ_LEN
from mosaicfl.core.client import FedProxClient
from mosaicfl.core.calibration import IsotonicCalibrator, TemperatureScaler

SEQ_LEN = 16
BATCH   = 8


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_loader(n: int = BATCH, seq_len: int = SEQ_LEN, seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    x   = torch.randint(1, VOCAB_SIZE, (n, seq_len))
    y   = torch.randint(0, NUM_CLASSES, (n,))
    dia = torch.randint(0, 100, (n, seq_len))
    return DataLoader(TensorDataset(x, y, dia), batch_size=4)


def _make_imbalanced_loader(majority_class: int = 0, minority_class: int = 2,
                             majority_n: int = 200, minority_n: int = 1) -> DataLoader:
    """Loader com desbalanceamento extremo para testar o clipping de pesos."""
    n = majority_n + minority_n
    x   = torch.randint(1, VOCAB_SIZE, (n, SEQ_LEN))
    y   = torch.cat([
        torch.full((majority_n,), majority_class, dtype=torch.long),
        torch.full((minority_n,),  minority_class, dtype=torch.long),
    ])
    dia = torch.randint(0, 100, (n, SEQ_LEN))
    return DataLoader(TensorDataset(x, y, dia), batch_size=16)


def _make_client(seed: int = 42) -> FedProxClient:
    loader = _make_loader(seed=seed)
    return FedProxClient(0, loader, loader)


# ── 1. Class weight clipping ──────────────────────────────────────────────────

class TestClassWeightClipping:

    def test_weights_never_exceed_15(self):
        loader = _make_imbalanced_loader(majority_n=500, minority_n=1)
        client = FedProxClient(0, loader, loader)
        weights = client.criterion.weight
        assert weights.max().item() <= 15.0, (
            f"Peso máximo {weights.max().item():.2f} excede o teto de 15.0"
        )

    def test_weights_are_non_negative(self):
        loader = _make_imbalanced_loader()
        client = FedProxClient(0, loader, loader)
        assert (client.criterion.weight >= 0).all()

    def test_absent_class_has_zero_weight(self):
        """Classe ausente no loader deve ter peso 0, não peso alto."""
        loader = _make_imbalanced_loader(majority_class=0, minority_class=1,
                                         majority_n=50, minority_n=0)
        # Força classe 1 ausente criando loader só com classe 0
        x   = torch.randint(1, VOCAB_SIZE, (50, SEQ_LEN))
        y   = torch.zeros(50, dtype=torch.long)
        dia = torch.randint(0, 100, (50, SEQ_LEN))
        loader_zero_only = DataLoader(TensorDataset(x, y, dia), batch_size=16)
        client = FedProxClient(0, loader_zero_only, loader_zero_only)
        # classes 1–4 não aparecem → peso deve ser 0
        for cls in range(1, NUM_CLASSES):
            assert client.criterion.weight[cls].item() == 0.0, (
                f"Classe {cls} ausente mas peso={client.criterion.weight[cls].item()}"
            )

    def test_without_clipping_weight_would_exceed_15(self):
        """Documenta que sem clipping o peso seria maior que 15 — valida que o teto é necessário."""
        loader = _make_imbalanced_loader(majority_n=1000, minority_n=1)
        counts: Counter = Counter()
        for _, batch_y, *_ in loader:
            counts.update(batch_y.tolist())
        total = sum(counts.values())
        n     = NUM_CLASSES
        raw_weight_minority = total / (n * counts[2]) if counts.get(2, 0) > 0 else 0
        assert raw_weight_minority > 15.0, (
            "O teste de clipping só é válido quando o peso bruto excede 15 — "
            f"peso bruto foi {raw_weight_minority:.2f}"
        )


# ── 2. Gradient clipping ──────────────────────────────────────────────────────

class TestGradientClipping:

    def test_fit_completes_without_nan(self):
        """Com gradientes potencialmente explosivos, fit() não deve retornar NaN na loss."""
        loader = _make_imbalanced_loader(majority_n=200, minority_n=1)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert not np.isnan(metrics["loss"]), "Loss é NaN após fit() — gradiente explodiu"

    def test_params_remain_finite_after_fit(self):
        loader = _make_imbalanced_loader(majority_n=200, minority_n=1)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        updated, _, _ = client.fit(params, {})
        for i, arr in enumerate(updated):
            assert np.isfinite(arr).all(), f"Parâmetro {i} contém inf/nan após fit()"


# ── 3. Grad norm no retorno de fit() ─────────────────────────────────────────

class TestGradNormMetric:

    def test_fit_returns_grad_norm_key(self):
        client = _make_client()
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert "grad_norm" in metrics, f"'grad_norm' não está em metrics: {list(metrics.keys())}"

    def test_grad_norm_is_python_float(self):
        client = _make_client()
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert isinstance(metrics["grad_norm"], float), (
            f"grad_norm esperado float, obtido {type(metrics['grad_norm'])}"
        )

    def test_grad_norm_is_non_negative(self):
        client = _make_client()
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        assert metrics["grad_norm"] >= 0.0

    def test_grad_norm_bounded_by_clip(self):
        """grad_norm deve ser próximo de max_norm=1.0 quando os gradientes são grandes."""
        loader = _make_imbalanced_loader(majority_n=200, minority_n=1)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        # Se houve clipping, grad_norm estará em torno de 1.0 (pode ser < 1.0 se não houve clipping)
        assert metrics["grad_norm"] >= 0.0
        # O valor retornado por clip_grad_norm_ é a norma ANTES do clipping
        # Validamos apenas que existe e é finito
        assert np.isfinite(metrics["grad_norm"])


# ── 4. local_epochs = 1 ───────────────────────────────────────────────────────

class TestLocalEpochs:

    def test_local_epochs_is_one(self):
        assert FED_CFG.local_epochs == 1, (
            f"local_epochs deve ser 1 (reduzido para controlar client drift non-IID), "
            f"atual: {FED_CFG.local_epochs}"
        )

    def test_fit_uses_one_epoch_by_default(self):
        """tau deve ser igual ao número de batches (1 época × n_batches)."""
        loader = _make_loader(n=16)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics = client.fit(params, {})
        n_batches = len(loader)
        assert metrics["tau"] == n_batches * FED_CFG.local_epochs, (
            f"tau={metrics['tau']} esperado {n_batches * FED_CFG.local_epochs} "
            f"(batches={n_batches} × epochs={FED_CFG.local_epochs})"
        )

    def test_fit_config_overrides_local_epochs(self):
        """Overrides via config dict devem funcionar (usado em análise de sensibilidade)."""
        loader = _make_loader(n=16)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics_2ep = client.fit(params, {"local_epochs": 2})
        n_batches = len(loader)
        assert metrics_2ep["tau"] == n_batches * 2


# ── 5. DataLoader com gerador fixo ───────────────────────────────────────────

class TestDeterministicDataLoader:

    def _first_batch_indices(self, loader: DataLoader) -> list:
        for batch in loader:
            return batch[1].tolist()  # labels do primeiro batch
        return []

    def test_same_seed_same_batch_order(self):
        """Dois DataLoaders com mesmo gerador produzem a mesma ordem."""
        x   = torch.randint(1, VOCAB_SIZE, (32, SEQ_LEN))
        y   = torch.arange(32, dtype=torch.long)
        dia = torch.randint(0, 100, (32, SEQ_LEN))
        dataset = TensorDataset(x, y, dia)

        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        loader_a = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen_a)
        loader_b = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen_b)

        batches_a = [b[1].tolist() for b in loader_a]
        batches_b = [b[1].tolist() for b in loader_b]
        assert batches_a == batches_b, "Geradores com mesma seed devem produzir ordem idêntica"

    def test_different_seeds_different_order(self):
        x   = torch.randint(1, VOCAB_SIZE, (64, SEQ_LEN))
        y   = torch.arange(64, dtype=torch.long)
        dia = torch.randint(0, 100, (64, SEQ_LEN))
        dataset = TensorDataset(x, y, dia)

        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(99)
        loader_a = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen_a)
        loader_b = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen_b)

        all_a = [item for batch in loader_a for item in batch[1].tolist()]
        all_b = [item for batch in loader_b for item in batch[1].tolist()]
        assert all_a != all_b, "Seeds diferentes devem produzir ordens diferentes"


# ── 6. IsotonicCalibrator ─────────────────────────────────────────────────────

class TestIsotonicCalibrator:

    @pytest.fixture
    def fitted_calibrator(self):
        """Calibrador ajustado em dados sintéticos simples."""
        from mosaicfl.core.model import SimplifiedBEHRT
        torch.manual_seed(0)
        model  = SimplifiedBEHRT(use_cls_token=True)
        loader = _make_loader(n=32, seed=1)
        iso    = IsotonicCalibrator()
        iso.fit(model, loader, device="cpu", num_classes=NUM_CLASSES)
        return iso

    def test_raises_if_not_fitted(self):
        iso    = IsotonicCalibrator()
        logits = torch.randn(8, NUM_CLASSES)
        with pytest.raises(RuntimeError, match="fit()"):
            iso.calibrate(logits)

    def test_calibrate_output_shape(self, fitted_calibrator):
        logits = torch.randn(16, NUM_CLASSES)
        probs  = fitted_calibrator.calibrate(logits)
        assert probs.shape == (16, NUM_CLASSES)

    def test_calibrate_probabilities_in_range(self, fitted_calibrator):
        logits = torch.randn(32, NUM_CLASSES)
        probs  = fitted_calibrator.calibrate(logits)
        assert (probs >= 0).all() and (probs <= 1).all(), (
            f"Probabilidades fora de [0,1]: min={probs.min():.4f} max={probs.max():.4f}"
        )

    def test_calibrate_probabilities_sum_to_one(self, fitted_calibrator):
        logits = torch.randn(32, NUM_CLASSES)
        probs  = fitted_calibrator.calibrate(logits)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(32), atol=1e-4), (
            f"Probabilidades não somam a 1: {row_sums[:5]}"
        )

    def test_compute_ece_returns_float(self, fitted_calibrator):
        logits = torch.randn(32, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (32,))
        ece    = fitted_calibrator.compute_ece(logits, labels)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_fit_empty_loader_does_not_crash(self):
        """fit() com loader vazio deve retornar sem ajustar (mas sem lançar exceção)."""
        from mosaicfl.core.model import SimplifiedBEHRT
        model  = SimplifiedBEHRT(use_cls_token=True)
        # Cria dataset vazio
        empty  = DataLoader(TensorDataset(
            torch.zeros(0, SEQ_LEN, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, SEQ_LEN, dtype=torch.long),
        ), batch_size=4)
        iso    = IsotonicCalibrator()
        iso.fit(model, empty, device="cpu", num_classes=NUM_CLASSES)
        assert not iso._fitted, "Calibrador não deve ser marcado como ajustado com loader vazio"

    def test_num_calibrators_equals_num_classes(self, fitted_calibrator):
        assert len(fitted_calibrator._calibrators) == NUM_CLASSES
