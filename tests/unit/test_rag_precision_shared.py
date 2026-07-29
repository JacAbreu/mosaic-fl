"""
Testes para mosaicfl.core.rag.precision.eval_precision_at_k() — extraída de
experiments/training/core/rag.py em 2026-07-28 pra ser compartilhada entre
Caminho A (test_loader centralizado) e Caminho B (val_loader local por
cliente, ver mosaicfl.core.client.FedProxClient.evaluate()). Mesma lógica,
só o loader/rag mudam de origem.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.rag.precision import eval_precision_at_k

_LABELS = ["curado_pronto", "curado_internado", "melhora_pronto"]
_VOCAB_INVERSE = {3: "WBC_ALTO", 4: "PCR_ALTO", 5: "SATURACAO_BAIXA"}


def _make_loader(sequences, labels):
    batch_x = torch.tensor(sequences)
    batch_y = torch.tensor(labels)
    batch_dia = torch.zeros_like(batch_x)
    return [(batch_x, batch_y, batch_dia)]


class TestEvalPrecisionAtK:
    def test_all_hits_gives_precision_1(self):
        rag = MagicMock()
        rag.retrieve.return_value = [
            {"metadata": {"desfecho": "curado_pronto"}},
            {"metadata": {"desfecho": "curado_pronto"}},
            {"metadata": {"desfecho": "curado_pronto"}},
        ]
        loader = _make_loader([[3, 4, 5]], [0])  # label_idx=0 -> curado_pronto
        result = eval_precision_at_k(rag, loader, _VOCAB_INVERSE, _LABELS, k=3)
        assert result["precision_at_3"] == 1.0
        assert result["n_queries"] == 1

    def test_all_misses_gives_precision_0(self):
        rag = MagicMock()
        rag.retrieve.return_value = [
            {"metadata": {"desfecho": "melhora_pronto"}},
            {"metadata": {"desfecho": "melhora_pronto"}},
            {"metadata": {"desfecho": "melhora_pronto"}},
        ]
        loader = _make_loader([[3, 4, 5]], [0])
        result = eval_precision_at_k(rag, loader, _VOCAB_INVERSE, _LABELS, k=3)
        assert result["precision_at_3"] == 0.0

    def test_partial_hits(self):
        rag = MagicMock()
        rag.retrieve.return_value = [
            {"metadata": {"desfecho": "curado_pronto"}},
            {"metadata": {"desfecho": "melhora_pronto"}},
            {"metadata": {"desfecho": "curado_pronto"}},
        ]
        loader = _make_loader([[3, 4, 5]], [0])
        result = eval_precision_at_k(rag, loader, _VOCAB_INVERSE, _LABELS, k=3)
        # eval_precision_at_k arredonda em 4 casas — compara contra o mesmo arredondamento.
        assert result["precision_at_3"] == round(2 / 3, 4)

    def test_empty_loader_returns_zero_without_crashing(self):
        rag = MagicMock()
        result = eval_precision_at_k(rag, [], _VOCAB_INVERSE, _LABELS, k=3)
        assert result["precision_at_3"] == 0.0
        assert result["n_queries"] == 0

    def test_sequence_with_no_known_tokens_is_skipped(self):
        rag = MagicMock()
        loader = _make_loader([[99, 98]], [0])  # tokens fora do vocab_inverse
        result = eval_precision_at_k(rag, loader, _VOCAB_INVERSE, _LABELS, k=3)
        assert result["n_queries"] == 0
        rag.retrieve.assert_not_called()

    def test_per_class_precision_tracks_ground_truth_class(self):
        rag = MagicMock()
        rag.retrieve.return_value = [{"metadata": {"desfecho": "curado_internado"}}] * 3
        loader = _make_loader([[3, 4, 5]], [1])  # label_idx=1 -> curado_internado
        result = eval_precision_at_k(rag, loader, _VOCAB_INVERSE, _LABELS, k=3)
        assert result["per_class_precision_at_3"]["curado_internado"] == 1.0
        assert result["per_class_precision_at_3"]["curado_pronto"] is None
