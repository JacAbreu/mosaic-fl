"""
Testes para os utilitários de ECE federado em mosaicfl.core.evaluation —
local_ece_bin_stats(), merge_ece_bin_stats(), compute_ece_from_bin_totals().

Achado 2026-07-26: ece/ece_pre ficavam sempre None/0 em produção (Caminho B)
porque o único cálculo existente (calibration_mixin.py::_run_calibration) exige
um test_loader centralizado, indisponível por design de privacidade. Estes
utilitários permitem calcular o ECE global EXATO a partir de estatísticas por
bin já agregadas em cada hospital — nunca uma predição/rótulo bruto sai do
hospital.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.evaluation import (
    compute_ece,
    compute_ece_from_bin_totals,
    local_ece_bin_stats,
    merge_ece_bin_stats,
)


class TestLocalEceBinStats:
    def test_returns_fixed_length_list(self):
        confidences = torch.tensor([0.1, 0.5, 0.9])
        correct = torch.tensor([True, False, True])
        stats = local_ece_bin_stats(confidences, correct, n_bins=15)
        assert len(stats) == 15

    def test_counts_sum_to_total_samples(self):
        confidences = torch.rand(50)
        correct = torch.randint(0, 2, (50,)).bool()
        stats = local_ece_bin_stats(confidences, correct, n_bins=15)
        assert sum(b["count"] for b in stats) == 50

    def test_empty_bins_have_zero_sums(self):
        confidences = torch.tensor([0.05])  # só o 1º bin ocupado
        correct = torch.tensor([True])
        stats = local_ece_bin_stats(confidences, correct, n_bins=15)
        assert stats[0]["count"] == 1
        assert all(b["count"] == 0 for b in stats[1:])
        assert all(b["sum_confidence"] == 0.0 for b in stats[1:])

    def test_no_samples_all_bins_empty(self):
        stats = local_ece_bin_stats(torch.tensor([]), torch.tensor([]), n_bins=15)
        assert all(b["count"] == 0 for b in stats)


class TestMergeEceBinStats:
    def test_sums_counts_across_clients(self):
        client_a = local_ece_bin_stats(torch.tensor([0.9, 0.9]), torch.tensor([True, True]))
        client_b = local_ece_bin_stats(torch.tensor([0.9]), torch.tensor([False]))
        merged = merge_ece_bin_stats([client_a, client_b])
        total_count = sum(b["count"] for b in merged)
        assert total_count == 3

    def test_empty_list_returns_empty(self):
        assert merge_ece_bin_stats([]) == []

    def test_single_client_equals_itself(self):
        client_a = local_ece_bin_stats(torch.rand(20), torch.randint(0, 2, (20,)).bool())
        merged = merge_ece_bin_stats([client_a])
        assert merged == client_a


class TestComputeEceFromBinTotals:
    def test_zero_samples_returns_zero(self):
        empty_bins = [{"count": 0, "sum_confidence": 0.0, "sum_correct": 0.0} for _ in range(15)]
        ece, mce, n = compute_ece_from_bin_totals(empty_bins)
        assert (ece, mce, n) == (0.0, 0.0, 0)

    def test_perfectly_calibrated_has_zero_ece(self):
        """confidence=1.0 e 100% de acerto — mean_confidence == accuracy, gap=0
        exato (confidence=0.99 com 100% de acerto NÃO seria gap 0: seria 0.01,
        já que o modelo estaria subconfiante em relação à própria acurácia real)."""
        confidences = torch.full((100,), 1.0)
        correct = torch.ones(100).bool()
        bins = local_ece_bin_stats(confidences, correct)
        ece, mce, n = compute_ece_from_bin_totals(bins)
        assert ece == 0.0
        assert mce == 0.0
        assert n == 100

    def test_matches_centralized_compute_ece(self):
        """O ECE federado (soma de bins + cálculo final) tem que bater exatamente
        com compute_ece() rodado sobre o mesmo dado centralizado — é a garantia
        central de que a decomposição federada não perde precisão."""
        torch.manual_seed(0)
        confidences = torch.rand(200)
        correct = torch.randint(0, 2, (200,)).bool()

        centralized = compute_ece(confidences, correct)
        bins = local_ece_bin_stats(confidences, correct)
        ece, mce, n = compute_ece_from_bin_totals(bins)

        assert ece == centralized.ece
        assert mce == centralized.mce
        assert n == centralized.n_samples

    def test_two_client_split_matches_single_client_total(self):
        """Dividir a mesma população em 2 'clientes' e agregar tem que dar o mesmo
        resultado que calcular tudo de uma vez — é a prova de que a soma por bin
        não introduz nenhuma aproximação."""
        torch.manual_seed(1)
        confidences = torch.rand(300)
        correct = torch.randint(0, 2, (300,)).bool()

        centralized = compute_ece(confidences, correct)

        client_a_bins = local_ece_bin_stats(confidences[:120], correct[:120])
        client_b_bins = local_ece_bin_stats(confidences[120:], correct[120:])
        merged = merge_ece_bin_stats([client_a_bins, client_b_bins])
        ece, mce, n = compute_ece_from_bin_totals(merged)

        assert ece == centralized.ece
        assert mce == centralized.mce
        assert n == centralized.n_samples
