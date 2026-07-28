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

    SEQ_LEN = 16

    @pytest.fixture
    def dummy_loader(self):
        x   = torch.randint(1, VOCAB_SIZE, (8, self.SEQ_LEN))
        y   = torch.randint(0, NUM_CLASSES, (8,))
        dia = torch.randint(0, 100, (8, self.SEQ_LEN))
        return DataLoader(TensorDataset(x, y, dia), batch_size=4)

    @pytest.fixture
    def client_v2(self, dummy_loader):
        from mosaicfl.core.client import FedProxClient
        return FedProxClient(0, dummy_loader, dummy_loader)

    @pytest.fixture(scope="class")
    def contract_client(self):
        from mosaicfl.core.client import FedProxClient
        seq_len = 16
        x_tr   = torch.randint(1, VOCAB_SIZE, (self.TRAIN_SIZE, seq_len))
        y_tr   = torch.randint(0, NUM_CLASSES, (self.TRAIN_SIZE,))
        dia_tr = torch.randint(0, 100, (self.TRAIN_SIZE, seq_len))
        x_va   = torch.randint(1, VOCAB_SIZE, (self.VAL_SIZE, seq_len))
        y_va   = torch.randint(0, NUM_CLASSES, (self.VAL_SIZE,))
        dia_va = torch.randint(0, 100, (self.VAL_SIZE, seq_len))
        train_loader = DataLoader(TensorDataset(x_tr, y_tr, dia_tr), batch_size=4)
        val_loader   = DataLoader(TensorDataset(x_va, y_va, dia_va), batch_size=4)
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
        assert torch.isclose(client_v2._proximal_loss(loss, proximal_mu=0.1), loss)

    def test_proximal_loss_with_global_params_increases(self, client_v2):
        params = client_v2.get_parameters({})
        client_v2.set_parameters(params)
        for p in client_v2.model.parameters():
            p.data += 1.0
        loss = torch.tensor(1.0)
        result = client_v2._proximal_loss(loss, proximal_mu=0.1)
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
        from mosaicfl.core.client import FedProxClient
        x   = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        y   = torch.randint(0, NUM_CLASSES, (4,))
        dia = torch.randint(0, 100, (4, MAX_SEQ_LEN))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        updated, n, metrics = client.fit(params, {})
        assert isinstance(updated, list)

    def test_create_client_fn_factory(self):
        from mosaicfl.core.client import create_client_fn, FedProxClient
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

    # ── calibração federada (client-side fit, 2026-07-12) ────────────────────

    def test_evaluate_without_calibrate_flag_has_no_calibration_keys(self, contract_client):
        """Comportamento padrão (config vazio ou calibrate=False): sem overhead,
        sem chaves de calibração — mesmo contrato de antes desta funcionalidade."""
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(params, {})
        assert "calibration_method" not in metrics
        assert "temperature" not in metrics
        assert "isotonic_thresholds_json" not in metrics

    def test_evaluate_calibrate_temperature(self, contract_client):
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(
            params, {"calibrate": True, "calibration_method": "temperature"}
        )
        assert metrics["calibration_method"] == "temperature"
        assert isinstance(metrics["temperature"], float)
        assert metrics["temperature"] > 0

    def test_evaluate_calibrate_isotonic(self, contract_client):
        import json
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(
            params, {"calibrate": True, "calibration_method": "isotonic"}
        )
        assert metrics["calibration_method"] == "isotonic"
        thresholds = json.loads(metrics["isotonic_thresholds_json"])
        assert len(thresholds) == NUM_CLASSES
        for x_list, y_list in thresholds:
            assert len(x_list) == len(y_list)

    def test_evaluate_calibrate_defaults_to_temperature_method(self, contract_client):
        """calibration_method ausente na config, mas calibrate=True — não deve travar,
        cai no default 'temperature' (mesmo default de FED_CFG.calibration_method)."""
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(params, {"calibrate": True})
        assert metrics["calibration_method"] == "temperature"

    def test_fit_local_calibrator_isotonic_direct(self, client_v2):
        """Chama _fit_local_calibrator() diretamente com logits/labels sintéticos —
        evita depender do forward pass completo pra testar só a serialização."""
        import json
        logits = torch.randn(20, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (20,))
        result = client_v2._fit_local_calibrator("isotonic", logits, labels)
        assert result["calibration_method"] == "isotonic"
        thresholds = json.loads(result["isotonic_thresholds_json"])
        assert len(thresholds) == NUM_CLASSES

    def test_fit_local_calibrator_temperature_direct(self, client_v2):
        logits = torch.randn(20, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (20,))
        result = client_v2._fit_local_calibrator("temperature", logits, labels)
        assert result["calibration_method"] == "temperature"
        assert result["temperature"] > 0

    # ── ece/ece_pre federado (2026-07-26) ─────────────────────────────────────

    def test_evaluate_calibrate_includes_ece_pre_bin_stats(self, contract_client):
        import json
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(
            params, {"calibrate": True, "calibration_method": "temperature"}
        )
        assert "ece_pre_bin_stats_json" in metrics
        bins = json.loads(metrics["ece_pre_bin_stats_json"])
        assert len(bins) == 15
        assert all({"count", "sum_confidence", "sum_correct"} == set(b) for b in bins)
        assert sum(b["count"] for b in bins) == self.VAL_SIZE

    def test_evaluate_without_calibrate_flag_has_no_ece_keys(self, contract_client):
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(params, {})
        assert "ece_pre_bin_stats_json" not in metrics

    def test_fit_local_calibrator_temperature_includes_ece_post(self, client_v2):
        import json
        logits = torch.randn(20, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (20,))
        result = client_v2._fit_local_calibrator("temperature", logits, labels)
        bins = json.loads(result["ece_post_bin_stats_json"])
        assert len(bins) == 15
        assert sum(b["count"] for b in bins) == 20

    def test_fit_local_calibrator_isotonic_includes_ece_post(self, client_v2):
        import json
        logits = torch.randn(20, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (20,))
        result = client_v2._fit_local_calibrator("isotonic", logits, labels)
        bins = json.loads(result["ece_post_bin_stats_json"])
        assert len(bins) == 15
        assert sum(b["count"] for b in bins) == 20

    # ── calibration_method="auto" real no cliente (2026-07-26) ────────────────
    # Até aqui, "auto" caía silenciosamente no ramo de temperature — nenhum
    # código tratava esse valor no lado cliente (só existia "auto" real do lado
    # servidor, e só com test_loader centralizado, indisponível no Caminho B).

    def test_auto_returns_one_of_the_two_methods(self, client_v2):
        logits = torch.randn(30, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (30,))
        result = client_v2._fit_local_calibrator("auto", logits, labels)
        assert result["calibration_method"] in ("temperature", "isotonic")

    def test_auto_includes_ece_post_bin_stats(self, client_v2):
        import json
        logits = torch.randn(30, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (30,))
        result = client_v2._fit_local_calibrator("auto", logits, labels)
        bins = json.loads(result["ece_post_bin_stats_json"])
        assert sum(b["count"] for b in bins) == 30

    def test_auto_picks_lower_ece_between_temperature_and_isotonic(self, client_v2, monkeypatch):
        """Confirma o critério de escolha chamando os dois métodos e comparando
        com o que _fit_local_calibrator("auto", ...) realmente devolveu."""
        logits = torch.randn(40, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (40,))

        from mosaicfl.core.evaluation import compute_ece_from_bin_totals
        import json as json_mod

        temp_result = client_v2._fit_local_calibrator("temperature", logits, labels)
        iso_result  = client_v2._fit_local_calibrator("isotonic", logits, labels)
        temp_ece, _, _ = compute_ece_from_bin_totals(json_mod.loads(temp_result["ece_post_bin_stats_json"]))
        iso_ece, _, _  = compute_ece_from_bin_totals(json_mod.loads(iso_result["ece_post_bin_stats_json"]))
        expected_method = "temperature" if temp_ece <= iso_ece else "isotonic"

        auto_result = client_v2._fit_local_calibrator("auto", logits, labels)
        assert auto_result["calibration_method"] == expected_method

    def test_evaluate_calibrate_auto_end_to_end(self, contract_client):
        params = contract_client.get_parameters({})
        _, _, metrics = contract_client.evaluate(
            params, {"calibrate": True, "calibration_method": "auto"}
        )
        assert metrics["calibration_method"] in ("temperature", "isotonic")
        assert "ece_pre_bin_stats_json" in metrics
        assert "ece_post_bin_stats_json" in metrics


class TestFedProxClientResourceMetrics:
    """FL_COLLECT_RESOURCE_METRICS (FED_CFG.collect_resource_metrics) — ligado por
    padrão (sustenta a análise de custo/provisionamento do TCC), mas parametrizável:
    quem roda o MOSAIC-FL fora deste contexto pode desligar sem afetar o treino em si.
    FED_CFG é um dataclass frozen — object.__setattr__ contorna isso só para o teste,
    sempre restaurado no finally para não vazar estado entre testes (singleton de módulo).
    """

    SEQ_LEN = 16

    @pytest.fixture
    def loader(self):
        x   = torch.randint(1, VOCAB_SIZE, (8, self.SEQ_LEN))
        y   = torch.randint(0, NUM_CLASSES, (8,))
        dia = torch.randint(0, 100, (8, self.SEQ_LEN))
        return DataLoader(TensorDataset(x, y, dia), batch_size=4)

    def test_fit_includes_resource_metrics_when_enabled(self, loader):
        from mosaicfl.core.client import FedProxClient
        from mosaicfl.core.config import FED_CFG
        original = FED_CFG.collect_resource_metrics
        object.__setattr__(FED_CFG, "collect_resource_metrics", True)
        try:
            client = FedProxClient(0, loader, loader)
            params = client.get_parameters({})
            _, _, metrics = client.fit(params, {})
            assert "resource_duration_s" in metrics
            assert "resource_cpu_pct" in metrics
            assert "resource_ram_mb" in metrics
            assert metrics["resource_duration_s"] >= 0.0
        finally:
            object.__setattr__(FED_CFG, "collect_resource_metrics", original)

    def test_fit_omits_resource_metrics_when_disabled(self, loader):
        from mosaicfl.core.client import FedProxClient
        from mosaicfl.core.config import FED_CFG
        original = FED_CFG.collect_resource_metrics
        object.__setattr__(FED_CFG, "collect_resource_metrics", False)
        try:
            client = FedProxClient(0, loader, loader)
            params = client.get_parameters({})
            _, _, metrics = client.fit(params, {})
            assert "resource_duration_s" not in metrics
            assert "resource_cpu_pct" not in metrics
            assert "resource_ram_mb" not in metrics
            assert "resource_gpu_power_w" not in metrics
            # loss/tau continuam presentes — desligar a coleta não afeta o resto do contrato
            assert "loss" in metrics
        finally:
            object.__setattr__(FED_CFG, "collect_resource_metrics", original)

    def test_fit_omits_gpu_keys_when_no_gpu_available(self, loader, monkeypatch):
        """Máquina sem GPU NVIDIA (ex.: notebook do HSL) — sample_gpu_power_w() retorna
        None, chaves resource_gpu_* devem ficar ausentes (nunca None — Flower Metrics
        só aceita escalares)."""
        from mosaicfl.core.client import FedProxClient
        from mosaicfl.core.config import FED_CFG
        monkeypatch.setattr("mosaicfl.core.client.sample_gpu_power_w", lambda: None)
        original = FED_CFG.collect_resource_metrics
        object.__setattr__(FED_CFG, "collect_resource_metrics", True)
        try:
            client = FedProxClient(0, loader, loader)
            params = client.get_parameters({})
            _, _, metrics = client.fit(params, {})
            assert "resource_duration_s" in metrics
            assert "resource_gpu_power_w" not in metrics
            assert "resource_gpu_energy_wh" not in metrics
        finally:
            object.__setattr__(FED_CFG, "collect_resource_metrics", original)


class TestMacroAucOvr:
    """_macro_auc_ovr() — AUC-ROC macro (one-vs-rest) calculado localmente no cliente.
    Achado 2026-07-26: fl_trainings.macro_auc/macro_f1 sempre NULL no Caminho B porque
    AUC nunca era calculado em lugar nenhum (diferente de F1, que já era calculado no
    cliente desde a implementação do RAG federado)."""

    def test_perfect_separation_gives_auc_one(self):
        from mosaicfl.core.client import _macro_auc_ovr
        # classe 0: alta prob pra classe 0; classe 1: alta prob pra classe 1 — separação perfeita
        probs = torch.tensor([
            [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8],
        ])
        labels = [0, 0, 1, 1]
        auc = _macro_auc_ovr(probs, labels, num_classes=2)
        assert auc == pytest.approx(1.0)

    def test_returns_none_when_all_classes_missing_or_degenerate(self):
        from mosaicfl.core.client import _macro_auc_ovr
        # só a classe 0 está presente (100% das amostras) — as duas categorias
        # (positivo/negativo) nunca coexistem pra nenhuma classe
        probs = torch.tensor([[0.9, 0.05, 0.05]] * 4)
        labels = [0, 0, 0, 0]
        auc = _macro_auc_ovr(probs, labels, num_classes=3)
        assert auc is None

    def test_skips_missing_classes_averages_only_valid_ones(self):
        from mosaicfl.core.client import _macro_auc_ovr
        # classe 2 nunca aparece localmente (cenário real: HSL sem melhora_pronto) —
        # não deve derrubar o cálculo pras classes 0 e 1, que têm as duas categorias
        probs = torch.tensor([
            [0.9, 0.05, 0.05], [0.8, 0.1, 0.1],
            [0.1, 0.85, 0.05], [0.2, 0.75, 0.05],
        ])
        labels = [0, 0, 1, 1]
        auc = _macro_auc_ovr(probs, labels, num_classes=3)
        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_never_raises_on_degenerate_input(self):
        from mosaicfl.core.client import _macro_auc_ovr
        probs = torch.full((3, 5), 0.2)
        labels = [0, 0, 0]
        auc = _macro_auc_ovr(probs, labels, num_classes=5)  # não deve levantar exceção
        assert auc is None


class TestEvaluateMacroAuc:
    def test_evaluate_includes_macro_auc_when_computable(self):
        """Rótulos cobrindo pelo menos 2 classes com as duas categorias presentes —
        macro_auc deve aparecer e estar em [0,1]."""
        from mosaicfl.core.client import FedProxClient
        seq_len = 8
        x   = torch.randint(1, VOCAB_SIZE, (12, seq_len))
        y   = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])
        dia = torch.randint(0, 100, (12, seq_len))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics = client.evaluate(params, {})
        assert "macro_auc" in metrics
        assert 0.0 <= metrics["macro_auc"] <= 1.0

    def test_evaluate_does_not_crash_with_degenerate_labels(self):
        from mosaicfl.core.client import FedProxClient
        seq_len = 8
        x = torch.randint(1, VOCAB_SIZE, (4, seq_len))
        y = torch.zeros(4, dtype=torch.long)  # todas as amostras são da mesma classe
        dia = torch.randint(0, 100, (4, seq_len))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        _, _, metrics = client.evaluate(params, {})  # não deve levantar exceção
        assert "macro_auc" not in metrics


class TestClassWeightOverridesIntegration:
    """FED_CFG.class_weight_overrides (Strategy pattern, mosaicfl.core.class_weighting)
    precisa chegar de fato no criterion do cliente — sem isso, a config fica sem efeito."""

    def test_override_reaches_criterion_weight(self, monkeypatch):
        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import FedConfig, MODEL_CFG

        custom_cfg = FedConfig(class_weight_overrides={"curado_internado": 12.0})
        monkeypatch.setattr(client_module, "FED_CFG", custom_cfg)
        # Isola do banco local real (2º nível de prioridade, ver
        # TestClassWeightOverridesPriorityOrder) — sem isso o teste fica refém de
        # qualquer override real gravado em clinical.fl_orchestration_config.
        monkeypatch.setattr(client_module, "load_local_overrides", lambda db_url: None)

        from mosaicfl.core.client import FedProxClient
        seq_len = 8
        x   = torch.randint(1, VOCAB_SIZE, (10, seq_len))
        y   = torch.randint(0, NUM_CLASSES, (10,))
        dia = torch.randint(0, 100, (10, seq_len))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        client = client_module.FedProxClient(0, loader, loader)

        idx = MODEL_CFG.class_labels.index("curado_internado")
        assert client.criterion.weight[idx].item() == pytest.approx(12.0)

    def test_no_overrides_preserves_legacy_behavior(self, monkeypatch):
        """Sem overrides (default), o peso continua vindo só da frequência local —
        nenhuma classe fica presa num valor fixo."""
        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import FedConfig

        default_cfg = FedConfig()
        monkeypatch.setattr(client_module, "FED_CFG", default_cfg)
        monkeypatch.setattr(client_module, "load_local_overrides", lambda db_url: None)

        from mosaicfl.core.client import FedProxClient
        seq_len = 8
        x   = torch.randint(1, VOCAB_SIZE, (10, seq_len))
        y   = torch.zeros(10, dtype=torch.long)  # só a classe 0 presente
        dia = torch.randint(0, 100, (10, seq_len))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        client = client_module.FedProxClient(0, loader, loader)

        # classe 0 (única presente) deve ter peso baixo (dominante local); demais 0.0 (ausentes)
        assert client.criterion.weight[0].item() > 0.0
        assert all(w == 0.0 for w in client.criterion.weight[1:].tolist())

    def test_round_config_takes_priority_over_env_fallback(self):
        """Caminho de produção real: loader_factory (não train_loader direto) — o
        override chega via config["class_weight_overrides_json"] na 1ª chamada de
        evaluate()/fit(), o mesmo canal que fit_config_mixin.py injeta a partir de
        clinical.fl_orchestration_config (migration 028). Precisa ganhar de
        FED_CFG.class_weight_overrides (env), não só coexistir com ele."""
        import json as _json

        from mosaicfl.core.client import FedProxClient
        from mosaicfl.core.config import MODEL_CFG

        seq_len = 8

        def _loader_factory(vocab_json):
            x   = torch.randint(1, VOCAB_SIZE, (10, seq_len))
            y   = torch.randint(0, NUM_CLASSES, (10,))
            dia = torch.randint(0, 100, (10, seq_len))
            loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
            return loader, loader

        client = FedProxClient(client_id=0, loader_factory=_loader_factory)
        idx = MODEL_CFG.class_labels.index("melhora_pronto")
        config = {
            "vocab_json": _json.dumps({"CLS": 0}),
            "class_weight_overrides_json": _json.dumps({"melhora_pronto": 8.0}),
        }

        params = client.get_parameters({})
        client.fit(params, config)

        assert client.criterion.weight[idx].item() == pytest.approx(8.0)

    def test_malformed_round_config_falls_back_without_raising(self, monkeypatch):
        """class_weight_overrides_json malformado no config da rodada não pode
        derrubar o treino do cliente — cai no fallback local e segue."""
        import json as _json

        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import FedConfig

        monkeypatch.setattr(client_module, "FED_CFG", FedConfig())  # sem overrides

        seq_len = 8

        def _loader_factory(vocab_json):
            x   = torch.randint(1, VOCAB_SIZE, (10, seq_len))
            y   = torch.randint(0, NUM_CLASSES, (10,))
            dia = torch.randint(0, 100, (10, seq_len))
            loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
            return loader, loader

        client = client_module.FedProxClient(client_id=0, loader_factory=_loader_factory)
        config = {
            "vocab_json": _json.dumps({"CLS": 0}),
            "class_weight_overrides_json": "{not valid json",
        }

        params = client.get_parameters({})
        client.fit(params, config)  # não deve levantar exceção

        assert client.criterion is not None


class TestClassWeightOverridesPriorityOrder:
    """3 fontes possíveis de override, em ordem: config da rodada (servidor,
    compartilhado) > banco local desta máquina > FED_CFG (env, último fallback).
    Ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14 —
    banco local é o que permite BPSP e HSL terem pesos DIFERENTES entre si."""

    def _loader_factory(self, vocab_json):
        seq_len = 8
        x   = torch.randint(1, VOCAB_SIZE, (10, seq_len))
        y   = torch.randint(0, NUM_CLASSES, (10,))
        dia = torch.randint(0, 100, (10, seq_len))
        loader = DataLoader(TensorDataset(x, y, dia), batch_size=4)
        return loader, loader

    def test_local_db_used_when_no_server_config(self, monkeypatch):
        import json as _json

        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import MODEL_CFG

        monkeypatch.setattr(
            client_module, "load_local_overrides",
            lambda db_url: {"melhora_pronto": 8.0},
        )

        client = client_module.FedProxClient(client_id=0, loader_factory=self._loader_factory)
        idx = MODEL_CFG.class_labels.index("melhora_pronto")
        config = {"vocab_json": _json.dumps({"CLS": 0})}  # sem class_weight_overrides_json

        params = client.get_parameters({})
        client.fit(params, config)

        assert client.criterion.weight[idx].item() == pytest.approx(8.0)

    def test_server_config_wins_over_local_db(self, monkeypatch):
        import json as _json

        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import MODEL_CFG

        monkeypatch.setattr(
            client_module, "load_local_overrides",
            lambda db_url: {"melhora_pronto": 8.0},
        )

        client = client_module.FedProxClient(client_id=0, loader_factory=self._loader_factory)
        idx = MODEL_CFG.class_labels.index("melhora_pronto")
        config = {
            "vocab_json": _json.dumps({"CLS": 0}),
            "class_weight_overrides_json": _json.dumps({"melhora_pronto": 3.0}),
        }

        params = client.get_parameters({})
        client.fit(params, config)

        assert client.criterion.weight[idx].item() == pytest.approx(3.0)

    def test_env_fallback_used_when_no_server_config_and_no_local_db(self, monkeypatch):
        import json as _json

        import mosaicfl.core.client as client_module
        from mosaicfl.core.config import FedConfig, MODEL_CFG

        monkeypatch.setattr(client_module, "load_local_overrides", lambda db_url: None)
        monkeypatch.setattr(
            client_module, "FED_CFG",
            FedConfig(class_weight_overrides={"melhora_pronto": 5.0}),
        )

        client = client_module.FedProxClient(client_id=0, loader_factory=self._loader_factory)
        idx = MODEL_CFG.class_labels.index("melhora_pronto")
        config = {"vocab_json": _json.dumps({"CLS": 0})}

        params = client.get_parameters({})
        client.fit(params, config)

        assert client.criterion.weight[idx].item() == pytest.approx(5.0)
