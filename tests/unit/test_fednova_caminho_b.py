"""
Testes para ProductionFedProxStrategy._aggregate_fednova e sua integração em
aggregate_fit() — achado 2026-08-11: até esta mudança, o Caminho B nunca
normalizava a agregação por passos efetivos (τ), mesmo o Caminho A usando
FedNova desde o Exp 12 (Seção~sec:fedprox-fednova-gap do rascunho do TCC).
Liga/desliga via FED_CFG.aggregation_strategy ("fedavg" padrão | "fednova").

Cobre a camada de INTEGRAÇÃO com o protocolo Flower (extração de τ/parâmetros/
num_examples de FitRes reais) — a matemática em si já é coberta isoladamente em
tests/unit/test_aggregate_fednova.py. Mesmo padrão de mocking de
tests/unit/test_dp_noise_caminho_b.py.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import flwr as fl
import numpy as np

from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.model import SimplifiedBEHRT
from infrastructure.mosaicfl_server.strategy.core import ProductionFedProxStrategy


def _make_strategy(tmp_path):
    strategy = ProductionFedProxStrategy.__new__(ProductionFedProxStrategy)
    strategy.global_model = SimplifiedBEHRT()
    strategy.vocab = {}
    strategy.tracker = ConvergenceTracker(threshold=0.005, patience=3)
    strategy.round_counter = 0
    strategy.should_stop = False
    strategy.on_round_complete = None
    strategy._state_store = None
    strategy._round_timer = None
    strategy._last_round_metrics = {}
    strategy._history_rounds = []
    strategy._history_accuracies = []
    strategy._history_losses = []
    strategy._history_f1_macros = []
    strategy._history_per_class_f1 = []
    strategy._gpu_energy_wh_total = 0.0
    strategy._gpu_power_samples = []
    strategy._peak_ram_mb = 0.0
    strategy._cpu_pct_samples = []
    strategy._last_round_resources_json = None
    strategy._training_id = 70
    strategy._checkpoint_store = MagicMock()
    strategy._num_rounds = 50
    strategy._training_completed = False
    strategy.CHECKPOINT_DIR = tmp_path / "checkpoints"
    strategy.LOG_DIR = tmp_path / "logs"
    strategy.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    strategy.LOG_DIR.mkdir(parents=True, exist_ok=True)
    strategy._rdp_accountant = None
    strategy._dp_epsilon_simple = None
    strategy._dp_last_group_multipliers = None
    from infrastructure.mosaicfl_server.state_store import TrainingState
    strategy._current_state = TrainingState()
    return strategy


def _fit_result(client_id, ndarrays, num_examples, tau):
    """Simula (ClientProxy, FitRes) — mesmo padrão de _evaluate_result() em
    test_per_client_f1_extraction.py, adaptado pros campos que fit() usa
    (parameters/num_examples, não só metrics)."""
    metrics = {"client_id": client_id, "tau": tau, "loss": 0.1}
    return (
        None,
        SimpleNamespace(
            parameters=fl.common.ndarrays_to_parameters(ndarrays),
            num_examples=num_examples,
            metrics=metrics,
        ),
    )


class TestAggregateFedNovaIntegration:
    def test_extrai_tau_parametros_e_num_examples_corretamente(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        global_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]

        # 2 clientes com deltas e τ claramente diferentes — mesmo padrão real
        # (BPSP com mais dado/passos que o HSL).
        client_a = [arr + 1.0 for arr in global_ndarrays]
        client_b = [arr + 2.0 for arr in global_ndarrays]
        results = [
            _fit_result(0, client_a, num_examples=550, tau=55),
            _fit_result(1, client_b, num_examples=100, tau=10),
        ]

        result_params = strategy._aggregate_fednova(global_ndarrays, results, server_round=1)

        assert result_params is not None
        result_ndarrays = fl.common.parameters_to_ndarrays(result_params)
        # resultado deve diferir dos pesos globais originais (algo foi agregado)
        assert any(not (a == b).all() for a, b in zip(global_ndarrays, result_ndarrays))
        # e deve ter o mesmo shape/quantidade de arrays que o modelo original
        assert len(result_ndarrays) == len(global_ndarrays)
        for a, b in zip(result_ndarrays, global_ndarrays):
            assert a.shape == b.shape

    def test_tau_ausente_usa_fallback_e_nao_quebra(self, tmp_path, caplog):
        strategy = _make_strategy(tmp_path)
        global_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        client = [arr + 1.0 for arr in global_ndarrays]

        # metrics SEM "tau" — simula um cliente com código antigo ou bug de envio
        result_no_tau = (
            None,
            SimpleNamespace(
                parameters=fl.common.ndarrays_to_parameters(client),
                num_examples=100,
                metrics={"client_id": 0, "loss": 0.1},  # sem "tau"
            ),
        )

        result_params = strategy._aggregate_fednova(
            global_ndarrays, [result_no_tau], server_round=1,
        )

        assert result_params is not None  # não quebrou

    def test_erro_na_agregacao_retorna_none_para_fallback_no_chamador(self, tmp_path):
        strategy = _make_strategy(tmp_path)
        global_ndarrays = [np.zeros(3)]
        # client_ndarrays com shape incompatível força erro dentro de aggregate_fednova
        client_bad_shape = [np.zeros(5)]
        result = _fit_result(0, client_bad_shape, num_examples=10, tau=1)

        result_params = strategy._aggregate_fednova(global_ndarrays, [result], server_round=1)

        assert result_params is None  # sinaliza fallback, nunca propaga exceção

    def test_um_cliente_so_equivale_ao_estado_dele(self, tmp_path):
        """Propriedade usada para justificar que o leave-one-out (training_id
        26/28) não precisa ser refeito com FedNova — Seção~sec:leave-one-out-
        caminho-b do rascunho do TCC."""
        strategy = _make_strategy(tmp_path)
        global_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        client_state = [arr + 3.0 for arr in global_ndarrays]
        result = _fit_result(0, client_state, num_examples=100, tau=37)

        result_params = strategy._aggregate_fednova(global_ndarrays, [result], server_round=1)
        result_ndarrays = fl.common.parameters_to_ndarrays(result_params)

        for a, b in zip(result_ndarrays, client_state):
            np.testing.assert_allclose(a, b, rtol=1e-5)


class TestAggregateFitFedNovaToggle:
    def test_fedavg_por_padrao_nao_chama_fednova(self, tmp_path, monkeypatch):
        """FED_CFG.aggregation_strategy="fedavg" (default) — comportamento
        idêntico ao histórico, _aggregate_fednova nunca é chamado."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(aggregation_strategy="fedavg"))

        strategy = _make_strategy(tmp_path)
        params = fl.common.ndarrays_to_parameters(
            [v.numpy() for v in strategy.global_model.state_dict().values()]
        )
        strategy._aggregate_fednova = MagicMock()

        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(params, {})):
            strategy.aggregate_fit(1, [], [])

        strategy._aggregate_fednova.assert_not_called()

    def test_fednova_ligado_substitui_parametros_agregados(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(aggregation_strategy="fednova"))

        strategy = _make_strategy(tmp_path)
        global_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        fedavg_params = fl.common.ndarrays_to_parameters(global_ndarrays)  # resultado "FedAvg" simulado (sem mudança)
        client_state = [arr + 5.0 for arr in global_ndarrays]
        results = [_fit_result(0, client_state, num_examples=100, tau=10)]

        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(fedavg_params, {})):
            strategy.aggregate_fit(1, results, [])

        loaded_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        # se FedNova rodou de verdade, os pesos carregados devem refletir o
        # cliente (único), não os pesos "FedAvg" inalterados que o mock devolveu
        for loaded, client in zip(loaded_ndarrays, client_state):
            np.testing.assert_allclose(loaded, client, rtol=1e-5)

    def test_fednova_com_erro_mantem_resultado_fedavg_como_fallback(self, tmp_path, monkeypatch):
        """Se _aggregate_fednova falhar (retorna None), aggregate_fit deve
        manter o resultado já computado por super().aggregate_fit() — nunca
        travar o treino por causa da agregação nova."""
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(aggregation_strategy="fednova"))

        strategy = _make_strategy(tmp_path)
        global_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        fedavg_params = fl.common.ndarrays_to_parameters(global_ndarrays)
        strategy._aggregate_fednova = MagicMock(return_value=None)

        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(fedavg_params, {})):
            strategy.aggregate_fit(1, [], [])

        loaded_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        for loaded, original in zip(loaded_ndarrays, global_ndarrays):
            np.testing.assert_allclose(loaded, original, rtol=1e-5)
