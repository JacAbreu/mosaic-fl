"""
Testes para ProductionFedProxStrategy — DP-FedAvg portado do Caminho A pro Caminho B
(achado 2026-07-28: "o que tem no Caminho A tem que ter no Caminho B", decisão da
autora). Antes desta mudança, ruído DP nunca era aplicado no Caminho B — só no
manual_loop.py de simulação (Caminho A). Cobre _apply_dp_noise_to_aggregated()
isoladamente e a integração em aggregate_fit() (liga/desliga via
FED_CFG.dp_noise_multiplier).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import flwr as fl

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


def _real_parameters():
    model = SimplifiedBEHRT()
    ndarrays = [v.numpy() for v in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(ndarrays), model


class TestApplyDpNoiseToAggregatedIsolated:
    def test_mutates_parameters_and_returns_epsilon(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=1.0, dp_max_grad_norm=1.0))

        strategy = _make_strategy(tmp_path)
        params, _ = _real_parameters()
        original_ndarrays = fl.common.parameters_to_ndarrays(params)

        result = strategy._apply_dp_noise_to_aggregated(params, server_round=1, n_clients=2)

        result_ndarrays = fl.common.parameters_to_ndarrays(result)
        assert any(
            not (a == b).all() for a, b in zip(original_ndarrays, result_ndarrays)
        )
        assert strategy._dp_epsilon_simple is not None
        assert strategy._dp_epsilon_simple > 0
        assert strategy._dp_last_group_multipliers == {"all": 1.0}

    def test_steps_rdp_accountant_once_per_protected_group(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(
            dp_noise_multiplier=1.0, dp_max_grad_norm=1.0,
            dp_noise_strategy="layer_group", dp_noise_head_scale=0.5,
        ))

        strategy = _make_strategy(tmp_path)
        strategy._rdp_accountant = MagicMock()
        params, _ = _real_parameters()

        strategy._apply_dp_noise_to_aggregated(params, server_round=1, n_clients=2)

        # excluded (pos_encoder) tem multiplicador 0.0 e não deve gerar .step()
        called_multipliers = {c.kwargs["noise_multiplier"] for c in strategy._rdp_accountant.step.call_args_list}
        assert 0.0 not in called_multipliers
        assert strategy._rdp_accountant.step.call_count >= 1

    def test_never_raises_falls_back_to_original_parameters(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=1.0, dp_max_grad_norm=1.0))

        strategy = _make_strategy(tmp_path)
        # global_model quebrado de propósito -> keys() vai levantar
        strategy.global_model = MagicMock()
        strategy.global_model.state_dict.side_effect = RuntimeError("boom")
        params, _ = _real_parameters()

        result = strategy._apply_dp_noise_to_aggregated(params, server_round=1, n_clients=2)

        assert result is params  # devolve os parâmetros originais, sem levantar exceção


class TestAggregateFitDpToggle:
    def test_dp_disabled_by_default_leaves_parameters_unchanged(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=0.0))

        strategy = _make_strategy(tmp_path)
        params, _ = _real_parameters()
        original_ndarrays = fl.common.parameters_to_ndarrays(params)

        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(params, {})):
            strategy.aggregate_fit(1, [], [])

        loaded_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        for a, b in zip(original_ndarrays, loaded_ndarrays):
            assert (a == b).all()

    def test_dp_enabled_applies_noise_before_loading_weights(self, tmp_path, monkeypatch):
        import infrastructure.mosaicfl_server.strategy.core as core_module
        from mosaicfl.core.config import FedConfig
        monkeypatch.setattr(core_module, "FED_CFG", FedConfig(dp_noise_multiplier=1.0, dp_max_grad_norm=1.0))

        strategy = _make_strategy(tmp_path)
        params, _ = _real_parameters()
        original_ndarrays = fl.common.parameters_to_ndarrays(params)

        with patch("flwr.server.strategy.FedProx.aggregate_fit", return_value=(params, {})):
            strategy.aggregate_fit(1, [], [])

        loaded_ndarrays = [v.numpy() for v in strategy.global_model.state_dict().values()]
        assert any(
            not (a == b).all() for a, b in zip(original_ndarrays, loaded_ndarrays)
        )
        assert strategy._dp_epsilon_simple is not None
