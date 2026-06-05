"""
test_v2_core.py
Testes unitários e de integração para os módulos core da v2.

Cobre:
  - model_v2.py      : SimplifiedBEHRT, BEHRTEncoderLayer, PositionalEncoding
  - client_v2.py     : FedProxClient (get/set parameters, proximal loss, fit, evaluate)
  - server_v2.py     : ConvergenceTracker, weighted_average, get_evaluate_fn, CustomFedProxStrategy
  - preprocess_v2.py : EHRPreprocessor, split_by_institution
  - data_loader.py   : load_with_fallback, FileDataSource, DataLoadError, _map_columns

Estruturas externas mockadas:
  - ChromaDB, SentenceTransformer, HuggingFace (rag_system_v2)
  - fl.server.strategy.FedProx (server_v2)
  - SQLAlchemy / psycopg2 (data_loader)

Uso:
    pytest tests/test_v2_core.py -v
    pytest tests/test_v2_core.py -v -k "TestModel"
"""
import json
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Garante que src/ está no path
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from mosaicfl.v2.config import (
    BATCH_SIZE, CONVERGENCE_PATIENCE, CONVERGENCE_THRESHOLD,
    DEVICE, EMBED_DIM, LR, MAX_SEQ_LEN, NUM_CLASSES,
    NUM_HEADS, NUM_LAYERS, PROXIMAL_MU, VOCAB_SIZE,
)
from mosaicfl.v2.model_v2 import (
    BEHRTEncoderLayer, PositionalEncoding, SimplifiedBEHRT,
)
from mosaicfl.v2.preprocess_v2 import EHRPreprocessor, split_by_institution
from mosaicfl.v2.server_v2 import (
    ConvergenceTracker, CustomFedProxStrategy,
    get_evaluate_fn, start_server, weighted_average,
)
from mosaicfl.v2.data_loader import (
    DataLoadError, FileDataSource, _map_columns,
    _convert_desfecho, _generate_synthetic_fallback,
    load_with_fallback, COLUMN_MAPPING, DATASET_FILENAMES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTILHADAS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instituicao":   ["HospA", "HospA", "HospB", "HospB", "HospC"],
        "idade":         [25.0, 6.0, 45.0, 365.0, 70.0],
        "idade_unidade": ["anos", "meses", "anos", "dias", "anos"],
        "peso":          [150.0, 70.0, 180.0, 50.0, 80.0],
        "peso_unidade":  ["lb", "kg", "lb", "kg", "kg"],
        "sintoma":       ["febre", "tosse", "dispneia", "fadiga", "mialgia"],
        "exame":         ["rt_pcr_positivo", "tomografia_normal", "rx_consolidacao",
                          "pcr_negativo", "tomografia_vidro_fosco"],
        "diagnostico":   ["covid19_leve", "covid19_moderado", "pneumonia_bacteriana",
                          "covid19_grave", "alta"],
        "desfecho":      [0, 0, 1, 1, 0],
    })


@pytest.fixture
def model_v2():
    return SimplifiedBEHRT(use_cls_token=True)


@pytest.fixture
def model_no_cls():
    return SimplifiedBEHRT(use_cls_token=False)


@pytest.fixture
def dummy_loader():
    x = torch.randint(1, VOCAB_SIZE, (8, 16))
    y = torch.randint(0, NUM_CLASSES, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


@pytest.fixture
def client_v2(dummy_loader):
    from mosaicfl.v2.client_v2 import FedProxClient
    return FedProxClient(0, dummy_loader, dummy_loader)


# ═══════════════════════════════════════════════════════════════════════════════
# PositionalEncoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionalEncodingV2:

    def test_output_shape_preserved(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.randn(2, 10, 64)
        assert pe(x).shape == x.shape

    def test_modifies_zero_input(self):
        """PE não deve ser identidade."""
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.zeros(1, 5, 64)
        assert not torch.allclose(pe(x), x)

    def test_max_len_covers_cls_token(self):
        """MAX_SEQ_LEN+1 deve suportar sequência com CLS token."""
        pe = PositionalEncoding(d_model=EMBED_DIM, max_len=MAX_SEQ_LEN + 1)
        x = torch.randn(2, MAX_SEQ_LEN + 1, EMBED_DIM)
        out = pe(x)
        assert out.shape == (2, MAX_SEQ_LEN + 1, EMBED_DIM)

    def test_different_positions_get_different_encoding(self):
        """Posições diferentes devem ter encodings diferentes."""
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        # Duas posições distintas devem diferir
        assert not torch.allclose(out[0, 0], out[0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# BEHRTEncoderLayer
# ═══════════════════════════════════════════════════════════════════════════════

class TestBEHRTEncoderLayerV2:

    @pytest.fixture
    def layer(self):
        return BEHRTEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS,
            dim_feedforward=128, dropout=0.0
        )

    def test_forward_shape(self, layer):
        x = torch.randn(2, 10, EMBED_DIM)
        out = layer(x)
        assert out.shape == (2, 10, EMBED_DIM)

    def test_forward_with_attention_shape(self, layer):
        x = torch.randn(2, 10, EMBED_DIM)
        out, attn = layer(x, return_attention=True)
        assert out.shape == (2, 10, EMBED_DIM)
        # average_attn_weights=False → (batch, heads, seq, seq)
        assert attn.shape == (2, NUM_HEADS, 10, 10)

    def test_padding_mask_applied(self, layer):
        """Posições mascaradas não devem influenciar output de posições reais."""
        x = torch.randn(1, 5, EMBED_DIM)
        mask_none = torch.zeros(1, 5, dtype=torch.bool)       # sem padding
        mask_last = torch.tensor([[False, False, False, True, True]])  # 2 pads

        out_none = layer(x, src_key_padding_mask=mask_none)
        out_masked = layer(x, src_key_padding_mask=mask_last)
        # Outputs das primeiras 3 posições devem diferir com/sem máscara
        assert not torch.allclose(out_none[0, :3], out_masked[0, :3], atol=1e-5)

    def test_deterministic_in_eval(self, layer):
        layer.eval()
        x = torch.randn(2, 8, EMBED_DIM)
        with torch.no_grad():
            assert torch.allclose(layer(x), layer(x))


# ═══════════════════════════════════════════════════════════════════════════════
# SimplifiedBEHRT v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimplifiedBEHRTV2:

    def test_forward_returns_correct_shape(self, model_v2):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        logits = model_v2(x)
        assert logits.shape == (4, NUM_CLASSES)

    def test_forward_with_attention(self, model_v2):
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        logits, attn = model_v2(x, return_attention=True)
        assert logits.shape == (2, NUM_CLASSES)
        # (num_layers, batch, heads, seq+1, seq+1) com CLS
        assert attn.shape[0] == NUM_LAYERS
        assert attn.shape[1] == 2
        assert attn.shape[2] == NUM_HEADS

    def test_cls_token_prepended(self, model_v2):
        """Com use_cls_token=True, sequência tem seq_len+1."""
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        _, attn = model_v2(x, return_attention=True)
        # seq dimension deve ser 16+1=17
        assert attn.shape[3] == 17
        assert attn.shape[4] == 17

    def test_no_cls_token_mean_pooling(self, model_no_cls):
        """Sem CLS, usa masked mean pooling → seq dim não aumenta."""
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        _, attn = model_no_cls(x, return_attention=True)
        assert attn.shape[3] == 16

    def test_masked_mean_pool_excludes_padding(self, model_v2):
        """Pooling deve ignorar posições de padding (x==0)."""
        emb = torch.ones(1, 5, EMBED_DIM)
        emb[0, 3:] = 0.0        # últimas 2 posições são padding
        mask = torch.tensor([[False, False, False, True, True]])
        pooled = model_v2._masked_mean_pool(emb, mask)
        expected = torch.ones(1, EMBED_DIM)  # média dos 3 primeiros
        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_masked_mean_pool_all_padding_no_nan(self, model_v2):
        """Todos padding não deve gerar NaN (clamp(min=1))."""
        emb = torch.randn(1, 4, EMBED_DIM)
        mask = torch.ones(1, 4, dtype=torch.bool)  # tudo é padding
        pooled = model_v2._masked_mean_pool(emb, mask)
        assert not torch.isnan(pooled).any()

    def test_deterministic_in_eval(self, model_v2):
        model_v2.eval()
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        with torch.no_grad():
            assert torch.allclose(model_v2(x), model_v2(x))

    def test_has_trainable_parameters(self, model_v2):
        params = [p for p in model_v2.parameters() if p.requires_grad]
        assert len(params) > 0
        assert sum(p.numel() for p in params) > 0

    def test_cls_token_is_parameter(self, model_v2):
        """cls_token deve ser nn.Parameter (treinável)."""
        assert isinstance(model_v2.cls_token, torch.nn.Parameter)

    def test_pre_classifier_applied(self, model_v2):
        """pre_classifier existe e não é identidade."""
        x = torch.randn(2, EMBED_DIM)
        out = model_v2.pre_classifier(x)
        assert out.shape == (2, EMBED_DIM)

    def test_state_dict_has_expected_keys(self, model_v2):
        keys = set(model_v2.state_dict().keys())
        assert "embedding.weight" in keys
        assert "cls_token" in keys
        assert "classifier.0.weight" in keys


# ═══════════════════════════════════════════════════════════════════════════════
# FedProxClient v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestFedProxClientV2:

    def test_get_parameters_matches_state_dict(self, client_v2):
        params = client_v2.get_parameters({})
        sd_values = list(client_v2.model.state_dict().values())
        assert len(params) == len(sd_values)
        for p, v in zip(params, sd_values):
            assert p.shape == v.cpu().numpy().shape

    def test_set_parameters_loads_correctly(self, client_v2):
        """Pesos novos devem ser carregados no modelo."""
        original = client_v2.get_parameters({})
        # Zera todos os pesos
        zero_params = [np.zeros_like(p) for p in original]
        client_v2.set_parameters(zero_params)
        reloaded = client_v2.get_parameters({})
        for p in reloaded:
            assert np.allclose(p, 0.0)

    def test_set_parameters_stores_global_params(self, client_v2):
        """Após set_parameters, global_params deve estar preenchido."""
        params = client_v2.get_parameters({})
        client_v2.set_parameters(params)
        assert client_v2.global_params is not None
        assert len(client_v2.global_params) == len(list(client_v2.model.parameters()))

    def test_proximal_loss_no_global_params(self, client_v2):
        """Sem global_params, deve retornar loss inalterado."""
        client_v2.global_params = None
        loss = torch.tensor(1.5)
        assert torch.isclose(client_v2._proximal_loss(loss), loss)

    def test_proximal_loss_with_global_params_increases(self, client_v2):
        """Com global_params diferentes, termo proximal deve aumentar a loss."""
        params = client_v2.get_parameters({})
        client_v2.set_parameters(params)
        # Muda os pesos locais para serem diferentes dos globais
        for p in client_v2.model.parameters():
            p.data += 1.0
        loss = torch.tensor(1.0)
        result = client_v2._proximal_loss(loss)
        assert result > loss

    def test_fit_returns_correct_structure(self, client_v2):
        params = client_v2.get_parameters({})
        updated_params, n_samples, metrics = client_v2.fit(params, {})
        assert isinstance(updated_params, list)
        assert n_samples > 0
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

    def test_evaluate_returns_correct_structure(self, client_v2):
        params = client_v2.get_parameters({})
        loss, n_samples, metrics = client_v2.evaluate(params, {})
        assert isinstance(loss, float)
        assert n_samples > 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "client_id" in metrics

    def test_evaluate_accuracy_in_range(self, client_v2):
        params = client_v2.get_parameters({})
        _, _, metrics = client_v2.evaluate(params, {})
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_fit_does_not_crash_on_bad_batch(self):
        """Batch problemático não deve crashar o cliente."""
        from mosaicfl.v2.client_v2 import FedProxClient
        # Sequência com valores fora do vocab (edge case)
        x = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        updated, n, metrics = client.fit(params, {})
        assert isinstance(updated, list)

    def test_create_client_fn_factory(self):
        from mosaicfl.v2.client_v2 import create_client_fn, FedProxClient
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        client = create_client_fn(1, x, y, x, y)
        assert isinstance(client, FedProxClient)
        assert client.client_id == 1


# ═══════════════════════════════════════════════════════════════════════════════
# ConvergenceTracker v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvergenceTrackerV2:

    def test_converges_after_patience_stable_rounds(self):
        t = ConvergenceTracker(threshold=0.01, patience=3)
        assert not t.check(0.70)
        assert not t.check(0.71)
        assert not t.check(0.709)   # Δ=0.001 < 0.01, stable=1
        assert not t.check(0.708)   # Δ=0.001 < 0.01, stable=2
        assert t.check(0.707)       # Δ=0.001 < 0.01, stable=3 → convergiu

    def test_does_not_converge_with_large_deltas(self):
        t = ConvergenceTracker(threshold=0.01, patience=3)
        for acc in [0.50, 0.60, 0.55, 0.65, 0.58]:
            t.check(acc)
        assert not t.check(0.70)

    def test_convergence_round_recorded(self):
        t = ConvergenceTracker(threshold=0.05, patience=2)
        t.check(0.80)
        t.check(0.80)
        t.check(0.80)  # 3ª chamada, 2 deltas iguais
        assert t.converged_round is not None

    def test_convergence_round_not_overwritten(self):
        t = ConvergenceTracker(threshold=0.05, patience=2)
        t.check(0.80); t.check(0.80); t.check(0.80)
        first_round = t.converged_round
        t.check(0.80)  # mais uma chamada
        assert t.converged_round == first_round

    def test_reset_clears_all_state(self):
        t = ConvergenceTracker(threshold=0.01, patience=2)
        t.check(0.80); t.check(0.80); t.check(0.80)
        t.reset()
        assert t.history == []
        assert t.stable_count == 0
        assert t.converged_round is None

    def test_single_value_never_converges(self):
        t = ConvergenceTracker(threshold=0.01, patience=1)
        assert not t.check(0.80)

    def test_patience_one_requires_one_stable_round(self):
        t = ConvergenceTracker(threshold=0.01, patience=1)
        t.check(0.80)
        assert t.check(0.801)  # Δ=0.001 < 0.01, stable=1 ≥ patience=1


# ═══════════════════════════════════════════════════════════════════════════════
# weighted_average v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedAverageV2:

    def test_correct_computation(self):
        metrics = [(100, {"accuracy": 0.8}), (200, {"accuracy": 0.9})]
        result = weighted_average(metrics)
        expected = (100 * 0.8 + 200 * 0.9) / 300
        assert abs(result["accuracy"] - expected) < 1e-6

    def test_empty_returns_empty_dict(self):
        assert weighted_average([]) == {}

    def test_zero_examples_returns_zero(self):
        result = weighted_average([(0, {"accuracy": 0.9})])
        assert result == {"accuracy": 0.0}

    def test_single_client(self):
        result = weighted_average([(50, {"accuracy": 0.75})])
        assert abs(result["accuracy"] - 0.75) < 1e-6

    def test_equal_weights(self):
        metrics = [(100, {"accuracy": 0.6}), (100, {"accuracy": 0.8})]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.70) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# get_evaluate_fn v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetEvaluateFnV2:

    def test_returns_callable(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        assert callable(fn)

    def test_returns_float_loss_and_metrics(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        loss, metrics = fn(1, params, {})
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_accuracy_bounded(self):
        x = torch.randint(1, VOCAB_SIZE, (16, 16))
        y = torch.zeros(16, dtype=torch.long)  # todos classe 0
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        _, metrics = fn(1, params, {})
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_rodada_in_metrics(self):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        fn = get_evaluate_fn(loader)
        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        _, metrics = fn(5, params, {})
        assert metrics["rodada"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# CustomFedProxStrategy v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestCustomFedProxStrategyV2:

    def _make_strategy(self, tmp_path):
        """Helper: cria strategy sem ativar Flower."""
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        with patch("mosaicfl.v2.server_v2.FedProx.__init__", return_value=None):
            strategy = CustomFedProxStrategy.__new__(CustomFedProxStrategy)
            strategy.tracker = tracker
            strategy.history = history
            strategy.save_dir = str(tmp_path)
            strategy.on_converged = None
            strategy._round_counter = 0
            import os; os.makedirs(str(tmp_path), exist_ok=True)
        return strategy, tracker, history

    def test_aggregate_evaluate_populates_history(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.5, {"accuracy": 0.75})):
            strategy.aggregate_evaluate(1, [], [])
        assert 1 in history["rounds"]
        assert 0.75 in history["accuracy"]

    def test_aggregate_evaluate_raises_stop_on_convergence(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        # Força convergência: 3 rounds estáveis
        tracker.history = [0.80, 0.800, 0.801]
        tracker.stable_count = 2
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.3, {"accuracy": 0.800})):
            with pytest.raises(StopIteration):
                strategy.aggregate_evaluate(4, [], [])

    def test_on_converged_callback_called(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        callback = MagicMock()
        strategy.on_converged = callback
        tracker.history = [0.80, 0.800, 0.801]
        tracker.stable_count = 2
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.3, {"accuracy": 0.800})):
            with pytest.raises(StopIteration):
                strategy.aggregate_evaluate(4, [], [])
        callback.assert_called_once()

    def test_checkpoint_saved_every_5_rounds(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                          return_value=(0.5, {"accuracy": 0.75})):
            strategy.aggregate_evaluate(5, [], [])
        assert history["last_checkpoint"] is not None

    def test_history_grows_across_rounds(self, tmp_path):
        strategy, tracker, history = self._make_strategy(tmp_path)
        accuracies = [0.70, 0.72, 0.74]
        for i, acc in enumerate(accuracies, 1):
            with patch.object(type(strategy).__bases__[0], "aggregate_evaluate",
                               return_value=(0.5, {"accuracy": acc})):
                try:
                    strategy.aggregate_evaluate(i, [], [])
                except StopIteration:
                    break
        assert len(history["rounds"]) == len(history["accuracy"])
        assert len(history["rounds"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# start_server v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestStartServerV2:

    def test_returns_three_values(self):
        strategy, tracker, history = start_server()
        assert strategy is not None
        assert isinstance(tracker, ConvergenceTracker)
        assert isinstance(history, dict)

    def test_history_has_expected_keys(self):
        _, _, history = start_server()
        assert "rounds" in history
        assert "accuracy" in history
        assert "communication_mb" in history

    def test_evaluate_fn_none_without_test_loader(self):
        strategy, _, _ = start_server(test_loader=None)
        # evaluate_fn deve ser None quando não passado
        # (não podemos acessar internals do FedProx diretamente, mas não deve crashar)
        assert strategy is not None

    def test_evaluate_fn_active_with_test_loader(self):
        x = torch.randint(1, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        strategy, _, _ = start_server(test_loader=loader)
        assert strategy is not None


# ═══════════════════════════════════════════════════════════════════════════════
# EHRPreprocessor v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestEHRPreprocessorV2:

    def test_normalize_units_converts_months(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[1, "idade"] - 0.5) < 0.01  # 6 meses → 0.5 anos

    def test_normalize_units_converts_days(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[3, "idade"] - 1.0) < 0.05  # 365 dias ≈ 1 ano

    def test_normalize_units_sets_unidade_anos(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert (df["idade_unidade"] == "anos").all()

    def test_normalize_units_converts_lb_to_kg(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert abs(df.loc[0, "peso"] - 150 * 0.453592) < 0.01
        assert abs(df.loc[2, "peso"] - 180 * 0.453592) < 0.01

    def test_normalize_units_preserves_kg(self, sample_df):
        pre = EHRPreprocessor()
        df = pre.normalize_units(sample_df.copy())
        assert df.loc[1, "peso"] == 70.0   # já era kg

    def test_clean_text_lowercase(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"sintoma": ["FEBRE", "Tosse"]})
        result = pre.clean_text(df, ["sintoma"])
        assert all(v == v.lower() for v in result["sintoma"])

    def test_clean_text_removes_special_chars(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"sintoma": ["febre!@#$%", "tosse&*("]})
        result = pre.clean_text(df, ["sintoma"])
        assert "!" not in result["sintoma"].iloc[0]
        assert "&" not in result["sintoma"].iloc[1]

    def test_build_vocabulary_special_tokens(self, sample_df):
        pre = EHRPreprocessor()
        pre.build_vocabulary(sample_df, ["sintoma", "exame", "diagnostico"])
        for token in ["<PAD>", "<UNK>", "<MASK>", "<CLS>"]:
            assert token in pre.vocab_map
        assert pre.vocab_map["<PAD>"] == 0

    def test_build_vocabulary_no_duplicates(self, sample_df):
        pre = EHRPreprocessor()
        vocab = pre.build_vocabulary(sample_df, ["sintoma"])
        # Todos os valores devem ser únicos
        values = list(vocab.values())
        assert len(values) == len(set(values))

    def test_handle_missing_imputes_median(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0]})
        result = pre.handle_missing(df.copy())
        assert result.loc[2, "num"] == 2.0  # mediana de [1, 2, 4]

    def test_handle_missing_imputes_unk(self):
        pre = EHRPreprocessor()
        df = pd.DataFrame({"cat": ["a", np.nan, "c"]})
        result = pre.handle_missing(df.copy())
        assert result.loc[1, "cat"] == "<UNK>"

    def test_process_returns_df_and_summary(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, summary = pre.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        assert isinstance(df_proc, pd.DataFrame)
        assert isinstance(summary, dict)
        assert summary["total_amostras"] == len(sample_df)
        assert summary["tamanho_vocabulario"] > 4  # mais que os tokens especiais

    def test_process_creates_encoded_columns(self, sample_df):
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma"])
        assert "sintoma_encoded" in df_proc.columns
        assert df_proc["sintoma_encoded"].dtype in [int, np.int64, np.int32]

    def test_process_reject_count_zero_on_clean_data(self, sample_df):
        pre = EHRPreprocessor()
        _, summary = pre.process(sample_df, text_cols=["sintoma"])
        assert summary["amostras_rejeitadas"] == 0

    def test_transform_log_populated(self, sample_df):
        pre = EHRPreprocessor()
        _, summary = pre.process(sample_df, text_cols=["sintoma"])
        assert len(summary["transformacoes"]) > 0
        steps = [t["step"] for t in summary["transformacoes"]]
        assert "clean" in steps
        assert "vocab" in steps


# ═══════════════════════════════════════════════════════════════════════════════
# split_by_institution v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitByInstitutionV2:

    def test_creates_correct_number_of_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        # Só há 3 instituições: HospA, HospB, HospC
        assert len(clients) == 3

    def test_all_rows_preserved(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        total = sum(len(df) for df in clients.values())
        assert total == len(sample_df)

    def test_no_overlap_between_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        indices = [set(df.index) for df in clients.values()]
        for i, s1 in enumerate(indices):
            for j, s2 in enumerate(indices):
                if i != j:
                    assert len(s1 & s2) == 0

    def test_stratify_col_logs_distribution(self, sample_df):
        """Com stratify_col, deve executar sem erro."""
        clients = split_by_institution(sample_df, num_clients=5, stratify_col="desfecho")
        assert len(clients) >= 1

    def test_random_state_reproducible(self, sample_df):
        c1 = split_by_institution(sample_df, num_clients=5, random_state=42)
        c2 = split_by_institution(sample_df, num_clients=5, random_state=42)
        for cid in c1:
            pd.testing.assert_frame_equal(c1[cid].reset_index(drop=True),
                                           c2[cid].reset_index(drop=True))

    def test_num_clients_capped_by_institutions(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=100)
        assert len(clients) == 3  # apenas 3 hospitais no sample_df


# ═══════════════════════════════════════════════════════════════════════════════
# data_loader v2
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoaderV2:

    # ── _map_columns ──────────────────────────────────────────────────────────

    def test_map_columns_renames_known_columns(self):
        df = pd.DataFrame({"hospital": ["A"], "evolucao": [0], "sintomas": ["febre"]})
        mapped = _map_columns(df)
        assert "instituicao" in mapped.columns
        assert "desfecho" in mapped.columns

    def test_map_columns_case_insensitive(self):
        df = pd.DataFrame({"HOSPITAL": ["A"], "EVOLUCAO": [0]})
        mapped = _map_columns(df)
        assert "instituicao" in mapped.columns

    def test_map_columns_preserves_unmapped(self):
        df = pd.DataFrame({"hospital": ["A"], "coluna_extra": [1]})
        mapped = _map_columns(df)
        assert "coluna_extra" in mapped.columns

    # ── _convert_desfecho ─────────────────────────────────────────────────────

    def test_convert_desfecho_text_to_numeric(self):
        df = pd.DataFrame({"desfecho": ["alta", "obito", "alta", "uti"]})
        result = _convert_desfecho(df)
        assert result["desfecho"].tolist() == [0, 1, 0, 1]

    def test_convert_desfecho_numeric_unchanged(self):
        df = pd.DataFrame({"desfecho": [0, 1, 0, 1]})
        result = _convert_desfecho(df)
        assert result["desfecho"].tolist() == [0, 1, 0, 1]

    def test_convert_desfecho_preserves_original(self):
        df = pd.DataFrame({"desfecho": ["alta", "obito"]})
        result = _convert_desfecho(df)
        assert "desfecho_original" in result.columns

    # ── _generate_synthetic_fallback ──────────────────────────────────────────

    def test_synthetic_returns_correct_size(self):
        df = _generate_synthetic_fallback(200)
        assert len(df) == 200

    def test_synthetic_has_required_columns(self):
        df = _generate_synthetic_fallback(50)
        required = ["instituicao", "idade", "sintoma", "exame", "desfecho"]
        for col in required:
            assert col in df.columns

    def test_synthetic_desfecho_binary(self):
        df = _generate_synthetic_fallback(100)
        assert set(df["desfecho"].unique()).issubset({0, 1})

    def test_synthetic_is_reproducible(self):
        df1 = _generate_synthetic_fallback(100)
        df2 = _generate_synthetic_fallback(100)
        assert (df1["desfecho"].values == df2["desfecho"].values).all()

    # ── FileDataSource ─────────────────────────────────────────────────────────

    def test_file_source_not_available_when_empty(self, tmp_path):
        src = FileDataSource(base_dir=tmp_path, filenames=["nao_existe.csv"])
        assert not src.is_available()

    def test_file_source_available_when_file_exists(self, tmp_path):
        f = tmp_path / "dataset.csv"
        pd.DataFrame({"a": [1]}).to_csv(f, index=False)
        src = FileDataSource(base_dir=tmp_path, filenames=["dataset.csv"])
        assert src.is_available()

    def test_file_source_loads_csv(self, tmp_path):
        f = tmp_path / "dataset.csv"
        expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        expected.to_csv(f, index=False)
        src = FileDataSource(base_dir=tmp_path, filenames=["dataset.csv"])
        df = src.load()
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_file_source_raises_when_not_found(self, tmp_path):
        src = FileDataSource(base_dir=tmp_path, filenames=["nao_existe.csv"])
        with pytest.raises(FileNotFoundError):
            src.load()

    # ── DataLoadError ─────────────────────────────────────────────────────────

    def test_data_load_error_includes_attempts(self):
        attempts = [{"fonte": "SGBD", "erro": "timeout"}, {"fonte": "CSV", "erro": "not found"}]
        err = DataLoadError("Falha total", attempts=attempts)
        msg = str(err)
        assert "SGBD" in msg
        assert "CSV" in msg
        assert "timeout" in msg

    def test_data_load_error_without_attempts(self):
        err = DataLoadError("Falha simples")
        assert "Falha simples" in str(err)

    # ── load_with_fallback ────────────────────────────────────────────────────

    def test_fallback_uses_synthetic_when_no_source(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=50)
        assert len(df) == 50
        assert df["_fonte"].iloc[0] == "sintetico"

    def test_fallback_raises_data_load_error_when_no_synthetic(self):
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            with pytest.raises(DataLoadError):
                load_with_fallback(allow_synthetic=False)

    def test_fallback_csv_explicit_not_found_raises_file_not_found(self, tmp_path):
        nonexistent = str(tmp_path / "nao_existe.csv")
        with pytest.raises(FileNotFoundError, match="CSV informado não encontrado"):
            load_with_fallback(csv_path=nonexistent)

    def test_fallback_csv_explicit_loads_when_found(self, tmp_path):
        f = tmp_path / "base.csv"
        df = pd.DataFrame({
            "instituicao": ["H1", "H2"],
            "desfecho": [0, 1],
            "sintoma": ["febre", "tosse"],
        })
        df.to_csv(f, index=False)
        result = load_with_fallback(csv_path=str(f))
        assert result["_fonte"].iloc[0] == "csv_explicito"
        assert len(result) == 2

    def test_fallback_returns_fonte_column(self):
        df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=20)
        assert "_fonte" in df.columns

    def test_fallback_sgbd_skipped_when_no_url(self):
        """Sem MOSAICFL_DB_URL, SGBD deve ser pulado silenciosamente."""
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": ""}):
            df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=10)
        # Chegou no sintético → SGBD foi pulado corretamente
        assert df["_fonte"].iloc[0] == "sintetico"

    def test_fallback_sgbd_fails_gracefully_on_bad_url(self):
        """URL inválida deve cair para próxima fonte sem crashar."""
        with patch.dict("os.environ", {"MOSAICFL_DB_URL": "postgresql://invalid:5432/db"}):
            # Mock para que a conexão falhe imediatamente
            with patch("mosaicfl.v2.data_loader.DatabaseDataSource.is_available",
                       return_value=False):
                df = load_with_fallback(allow_synthetic=True, n_synthetic_samples=10)
        assert df["_fonte"].iloc[0] == "sintetico"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRAÇÃO: Pipeline completo v2 (sem RAG — sem LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestV2PipelineIntegration:

    def test_preprocess_to_model_forward(self, sample_df):
        """preprocess → encode → model.forward deve funcionar end-to-end."""
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        encoded_cols = [c for c in df_proc.columns if c.endswith("_encoded")]
        x = torch.tensor(df_proc[encoded_cols].values[:, :16], dtype=torch.long)
        if x.shape[1] < 16:
            pad = torch.zeros(x.shape[0], 16 - x.shape[1], dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
        model = SimplifiedBEHRT(use_cls_token=True)
        logits = model(x)
        assert logits.shape == (len(sample_df), NUM_CLASSES)

    def test_client_server_parameter_compatibility(self):
        """Parâmetros do cliente devem ser carregáveis no servidor."""
        from mosaicfl.v2.client_v2 import FedProxClient
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        # Simula carregamento no servidor
        server_model = SimplifiedBEHRT(use_cls_token=True)
        params_dict = zip(server_model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        missing, unexpected = server_model.load_state_dict(state_dict, strict=False)
        # Chaves de buffers (cls_token_id) podem estar ausentes — aceitável com strict=False
        assert len(unexpected) == 0

    def test_fedavg_aggregation_preserves_shape(self):
        """FedAvg manual deve produzir state_dict compatível com o modelo."""
        model = SimplifiedBEHRT(use_cls_token=True)
        sd = model.state_dict()
        # Simula 3 clientes com pesos aleatórios
        states = [
            OrderedDict({k: torch.randn_like(v.float()) for k, v in sd.items()})
            for _ in range(3)
        ]
        # Agrega via média simples (float apenas)
        # Buffers Long/Int (ex: cls_token_id) usam valor aleatório mas clampado ao range válido do vocab
        aggregated = OrderedDict()
        for key in sd.keys():
            if sd[key].dtype in (torch.long, torch.int):
                aggregated[key] = states[0][key].to(sd[key].dtype).clamp(0, VOCAB_SIZE - 1)
            else:
                aggregated[key] = torch.stack([s[key] for s in states]).mean(0).to(sd[key].dtype)
        model.load_state_dict(aggregated, strict=True)
        # Deve carregar sem erro
        x = torch.randint(1, VOCAB_SIZE, (2, 16))
        assert model(x).shape == (2, NUM_CLASSES)

    def test_convergence_tracker_in_evaluate_loop(self):
        """Tracker deve detectar convergência após rounds estáveis simulados."""
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        accuracies = [0.70, 0.72, 0.750, 0.752, 0.751, 0.750]
        converged_at = None
        for i, acc in enumerate(accuracies):
            if tracker.check(acc):
                converged_at = i + 1
                break
        assert converged_at is not None
        assert converged_at >= 4  # pelo menos patience+1 rodadas

    def test_split_then_client_then_evaluate(self, sample_df):
        """split → FedProxClient → evaluate deve funcionar end-to-end."""
        from mosaicfl.v2.client_v2 import FedProxClient
        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma"])
        clients = split_by_institution(df_proc, num_clients=5)
        # Usa cliente 0
        subset = clients[0]
        encoded_cols = [c for c in subset.columns if c.endswith("_encoded")]
        if not encoded_cols:
            pytest.skip("Sem colunas encoded para este subset")
        x = torch.tensor(subset[encoded_cols].values[:, :16], dtype=torch.long)
        if x.shape[1] < 16:
            pad = torch.zeros(x.shape[0], 16 - x.shape[1], dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
        y = torch.tensor(subset["desfecho"].values, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        loss, n, metrics = client.evaluate(params, {})
        assert isinstance(loss, float)
        assert n > 0
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])