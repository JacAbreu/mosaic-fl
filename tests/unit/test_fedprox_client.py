import sys
import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.config import VOCAB_SIZE, NUM_CLASSES, MAX_SEQ_LEN
from mosaicfl.core.federated import weighted_average_loss, weighted_average_accuracy


class TestFedProxClient:
    """
    Testes de FedProxClient: comportamento geral + contrato estrito de fit() e evaluate().

    O contrato estrito verifica tipos Python exatos, shapes e compatibilidade com
    weighted_average_loss/accuracy. Se alguém renomear "loss"→"train_loss" ou retornar
    um tensor em vez de ndarray, um teste aqui quebra antes que o bug chegue ao servidor.
    """

    TRAIN_SIZE = 12
    VAL_SIZE = 8
    CONTRACT_CLIENT_ID = 42

    @pytest.fixture
    def dummy_loader(self):
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        return DataLoader(TensorDataset(x, y), batch_size=4)

    @pytest.fixture
    def client_v2(self, dummy_loader):
        from mosaicfl.core.client_v2 import FedProxClient
        return FedProxClient(0, dummy_loader, dummy_loader)

    @pytest.fixture(scope="class")
    def contract_client(self):
        from mosaicfl.core.client_v2 import FedProxClient
        x_tr = torch.randint(1, VOCAB_SIZE, (self.TRAIN_SIZE, 16))
        y_tr = torch.randint(0, NUM_CLASSES, (self.TRAIN_SIZE,))
        x_va = torch.randint(1, VOCAB_SIZE, (self.VAL_SIZE, 16))
        y_va = torch.randint(0, NUM_CLASSES, (self.VAL_SIZE,))
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=4)
        val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=4)
        return FedProxClient(client_id=self.CONTRACT_CLIENT_ID,
                             train_loader=train_loader, val_loader=val_loader)

    @pytest.fixture(scope="class")
    def fit_result(self, contract_client):
        params = contract_client.get_parameters({})
        return params, contract_client.fit(params, {})

    @pytest.fixture(scope="class")
    def evaluate_result(self, contract_client):
        params = contract_client.get_parameters({})
        return contract_client.evaluate(params, {})

    # ── testes gerais ─────────────────────────────────────────────────────────

    def test_get_parameters_matches_state_dict(self, client_v2):
        params = client_v2.get_parameters({})
        sd_values = list(client_v2.model.state_dict().values())
        assert len(params) == len(sd_values)
        for p, v in zip(params, sd_values):
            assert p.shape == v.cpu().numpy().shape

    def test_set_parameters_loads_correctly(self, client_v2):
        original = client_v2.get_parameters({})
        zero_params = [np.zeros_like(p) for p in original]
        client_v2.set_parameters(zero_params)
        reloaded = client_v2.get_parameters({})
        for p in reloaded:
            assert np.allclose(p, 0.0)

    def test_set_parameters_stores_global_params(self, client_v2):
        params = client_v2.get_parameters({})
        client_v2.set_parameters(params)
        assert client_v2.global_params is not None
        assert len(client_v2.global_params) == len(list(client_v2.model.parameters()))

    def test_proximal_loss_no_global_params(self, client_v2):
        client_v2.global_params = None
        loss = torch.tensor(1.5)
        assert torch.isclose(client_v2._proximal_loss(loss), loss)

    def test_proximal_loss_with_global_params_increases(self, client_v2):
        params = client_v2.get_parameters({})
        client_v2.set_parameters(params)
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

    def test_fit_does_not_crash_on_edge_case_batch(self):
        from mosaicfl.core.client_v2 import FedProxClient
        x = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        updated, n, metrics = client.fit(params, {})
        assert isinstance(updated, list)

    def test_create_client_fn_factory(self):
        from mosaicfl.core.client_v2 import create_client_fn, FedProxClient
        x = torch.randint(1, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        client = create_client_fn(1, x, y, x, y)
        assert isinstance(client, FedProxClient)
        assert client.client_id == 1

    # ── contrato de fit() ─────────────────────────────────────────────────────

    def test_fit_returns_three_elements(self, fit_result):
        _, result = fit_result
        assert len(result) == 3

    def test_fit_params_is_list(self, fit_result):
        _, (params, _, _) = fit_result
        assert type(params) is list

    def test_fit_each_param_is_ndarray(self, fit_result):
        _, (params, _, _) = fit_result
        for i, arr in enumerate(params):
            assert isinstance(arr, np.ndarray), f"tensor {i}: esperado np.ndarray, obtido {type(arr)}"

    def test_fit_floating_params_are_float32(self, fit_result):
        _, (params, _, _) = fit_result
        for i, arr in enumerate(params):
            if np.issubdtype(arr.dtype, np.floating):
                assert arr.dtype == np.float32, \
                    f"tensor {i}: peso flutuante esperado float32, obtido {arr.dtype}"

    def test_fit_n_samples_is_python_int(self, fit_result):
        _, (_, n_samples, _) = fit_result
        assert type(n_samples) is int, f"esperado int, obtido {type(n_samples)}"

    def test_fit_n_samples_equals_dataset_size(self, fit_result):
        _, (_, n_samples, _) = fit_result
        assert n_samples == self.TRAIN_SIZE

    def test_fit_metrics_contains_loss_key(self, fit_result):
        _, (_, _, metrics) = fit_result
        assert "loss" in metrics, f"chaves presentes: {list(metrics.keys())}"

    def test_fit_metrics_loss_is_python_float(self, fit_result):
        _, (_, _, metrics) = fit_result
        assert type(metrics["loss"]) is float, f"esperado float, obtido {type(metrics['loss'])}"

    def test_fit_metrics_loss_is_non_negative(self, fit_result):
        _, (_, _, metrics) = fit_result
        assert metrics["loss"] >= 0.0

    def test_fit_param_shapes_preserved(self, contract_client):
        params_in = contract_client.get_parameters({})
        params_out, _, _ = contract_client.fit(params_in, {})
        for i, (p_in, p_out) in enumerate(zip(params_in, params_out)):
            assert p_in.shape == p_out.shape, \
                f"shape do tensor {i} mudou: {p_in.shape} → {p_out.shape}"

    def test_fit_metrics_feed_to_weighted_average_loss(self, fit_result):
        _, (_, n_samples, metrics) = fit_result
        aggregated = weighted_average_loss([(n_samples, metrics)])
        assert "loss" in aggregated
        assert isinstance(aggregated["loss"], float)
        assert aggregated["loss"] >= 0.0

    def test_fit_renamed_key_silences_aggregation(self):
        """Documenta por que o contrato importa: chave errada → agregação silenciosa."""
        broken = [(100, {"train_loss": 0.42})]
        result = weighted_average_loss(broken)
        assert result == {"loss": 0.0}

    # ── contrato de evaluate() ────────────────────────────────────────────────

    def test_evaluate_returns_three_elements(self, evaluate_result):
        assert len(evaluate_result) == 3

    def test_evaluate_loss_is_python_float(self, evaluate_result):
        loss, _, _ = evaluate_result
        assert type(loss) is float, f"esperado float, obtido {type(loss)}"

    def test_evaluate_loss_is_non_negative(self, evaluate_result):
        loss, _, _ = evaluate_result
        assert loss >= 0.0

    def test_evaluate_n_samples_is_python_int(self, evaluate_result):
        _, n_samples, _ = evaluate_result
        assert type(n_samples) is int, f"esperado int, obtido {type(n_samples)}"

    def test_evaluate_n_samples_equals_dataset_size(self, evaluate_result):
        _, n_samples, _ = evaluate_result
        assert n_samples == self.VAL_SIZE

    def test_evaluate_metrics_contains_accuracy_key(self, evaluate_result):
        _, _, metrics = evaluate_result
        assert "accuracy" in metrics, f"chaves presentes: {list(metrics.keys())}"

    def test_evaluate_metrics_contains_client_id_key(self, evaluate_result):
        _, _, metrics = evaluate_result
        assert "client_id" in metrics, f"chaves presentes: {list(metrics.keys())}"

    def test_evaluate_metrics_accuracy_is_float(self, evaluate_result):
        _, _, metrics = evaluate_result
        assert isinstance(metrics["accuracy"], float), \
            f"esperado float, obtido {type(metrics['accuracy'])}"

    def test_evaluate_metrics_accuracy_in_range(self, evaluate_result):
        _, _, metrics = evaluate_result
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_metrics_client_id_matches_constructor(self, evaluate_result):
        _, _, metrics = evaluate_result
        assert metrics["client_id"] == self.CONTRACT_CLIENT_ID

    def test_evaluate_metrics_feed_to_weighted_average_accuracy(self, evaluate_result):
        _, n_samples, metrics = evaluate_result
        aggregated = weighted_average_accuracy([(n_samples, metrics)])
        assert "accuracy" in aggregated
        assert isinstance(aggregated["accuracy"], float)
        assert 0.0 <= aggregated["accuracy"] <= 1.0

    def test_evaluate_renamed_key_silences_aggregation(self):
        broken = [(100, {"acc": 0.82})]
        result = weighted_average_accuracy(broken)
        assert result == {"accuracy": 0.0}

    def test_evaluate_is_deterministic(self, contract_client):
        params = contract_client.get_parameters({})
        loss_a, _, _ = contract_client.evaluate(params, {})
        loss_b, _, _ = contract_client.evaluate(params, {})
        assert abs(loss_a - loss_b) < 1e-6
