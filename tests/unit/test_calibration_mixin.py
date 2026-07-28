"""
Testes para _CalibrationMixin._run_calibration — cobre a escolha entre os 3 modos de
FED_CFG.calibration_method ("temperature" | "isotonic" | "auto") e a persistência
correta do calibrador escolhido no checkpoint.

Os testes mockam _fit_temperature/_fit_isotonic (evitam treinar/avaliar um modelo real)
e focam na lógica de orquestração: qual branch é tomado, e o que é persistido.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.mosaicfl_server.strategy.calibration_mixin import _CalibrationMixin


def _make_strategy():
    """Instância mínima do mixin, sem passar por __init__ real (mesmo padrão de
    test_custom_fedprox_strategy.py)."""
    strategy = _CalibrationMixin.__new__(_CalibrationMixin)
    strategy.global_model = MagicMock()
    strategy._test_loader = MagicMock()  # não-None: habilita a calibração
    strategy._checkpoint_store = MagicMock()
    strategy.vocab = {"A": 2}
    strategy._last_round_metrics = {"accuracy": 0.7, "loss": 0.3}
    strategy._save_evaluation_report = MagicMock()
    # Pesos/rodada/accuracy da "melhor rodada" (achado 2026-07-28) — usados por
    # _run_calibration() em vez dos pesos da rodada em que ela foi disparada.
    strategy._best_round = 3
    strategy._best_accuracy = 0.7
    strategy._best_state_dict = None  # None = não troca, usa self.global_model como está
    return strategy


def _fake_report(ece: float):
    """EvaluationReport mínimo — só o que _run_calibration lê (calibration.ece/mce, macro_*)."""
    report = MagicMock()
    report.calibration.ece = ece
    report.calibration.mce = ece
    report.macro_auc = 0.8
    report.macro_f1 = 0.5
    return report


def _fit_result(method: str, ece: float, temperature: float = 1.0):
    return {
        "method":               method,
        "temperature":          temperature,
        "isotonic_calibrators": ["fake_ir"] * 5 if method == "isotonic" else None,
        "isotonic_num_classes": 5 if method == "isotonic" else 0,
        "report_cal":           _fake_report(ece),
    }


class TestNoTestLoader:
    def test_skips_when_test_loader_none(self):
        strategy = _make_strategy()
        strategy._test_loader = None
        strategy._run_calibration(server_round=5)
        strategy._checkpoint_store.save.assert_not_called()


class TestTemperatureMethod:
    def test_uses_temperature_branch(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "temperature"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.05, temperature=1.8))
            strategy._fit_isotonic    = MagicMock()

            strategy._run_calibration(server_round=3)

            strategy._fit_temperature.assert_called_once()
            strategy._fit_isotonic.assert_not_called()
            save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
            assert save_kwargs["calibration_method"] == "temperature"
            assert save_kwargs["temperature"] == pytest.approx(1.8)
            assert save_kwargs["isotonic_calibrators"] is None


class TestIsotonicMethod:
    def test_uses_isotonic_branch(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "isotonic"
            strategy._fit_temperature = MagicMock()
            strategy._fit_isotonic    = MagicMock(return_value=_fit_result("isotonic", 0.03))

            strategy._run_calibration(server_round=3)

            strategy._fit_isotonic.assert_called_once()
            strategy._fit_temperature.assert_not_called()
            save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
            assert save_kwargs["calibration_method"] == "isotonic"
            assert save_kwargs["isotonic_calibrators"] == ["fake_ir"] * 5
            assert save_kwargs["isotonic_num_classes"] == 5

    def test_isotonic_failure_aborts_without_saving(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "isotonic"
            strategy._fit_isotonic = MagicMock(return_value=None)

            strategy._run_calibration(server_round=3)

            strategy._checkpoint_store.save.assert_not_called()


class TestAutoMethod:
    def test_picks_isotonic_when_lower_ece(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "auto"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.08))
            strategy._fit_isotonic    = MagicMock(return_value=_fit_result("isotonic", 0.03))

            strategy._run_calibration(server_round=3)

            strategy._fit_temperature.assert_called_once()
            strategy._fit_isotonic.assert_called_once()
            save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
            assert save_kwargs["calibration_method"] == "isotonic"

    def test_picks_temperature_when_lower_ece(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "auto"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.02))
            strategy._fit_isotonic    = MagicMock(return_value=_fit_result("isotonic", 0.09))

            strategy._run_calibration(server_round=3)

            save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
            assert save_kwargs["calibration_method"] == "temperature"

    def test_falls_back_when_one_fails(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "auto"
            strategy._fit_temperature = MagicMock(return_value=None)
            strategy._fit_isotonic    = MagicMock(return_value=_fit_result("isotonic", 0.03))

            strategy._run_calibration(server_round=3)

            save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
            assert save_kwargs["calibration_method"] == "isotonic"

    def test_aborts_when_both_fail(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "auto"
            strategy._fit_temperature = MagicMock(return_value=None)
            strategy._fit_isotonic    = MagicMock(return_value=None)

            strategy._run_calibration(server_round=3)

            strategy._checkpoint_store.save.assert_not_called()


class TestUnknownMethodFallsBackToTemperature:
    def test_unknown_method_uses_temperature(self):
        strategy = _make_strategy()
        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "nonsense"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.05))
            strategy._fit_isotonic    = MagicMock()

            strategy._run_calibration(server_round=3)

            strategy._fit_temperature.assert_called_once()
            strategy._fit_isotonic.assert_not_called()


class TestUsesBestRoundNotCurrentRound:
    """Achado 2026-07-28 — mesma classe do bug de checkpoint já corrigido em
    aggregate_fit()/_persist_federated_calibration: _run_calibration() pode ser
    disparada bem depois da melhor rodada (converged fica permanente), então
    precisa avaliar/salvar com os pesos/round/accuracy da MELHOR rodada."""

    def test_loads_best_state_dict_before_evaluating(self):
        strategy = _make_strategy()
        fake_best = {"marker": "sou o melhor round"}
        strategy._best_state_dict = fake_best
        strategy._best_round = 42
        strategy._best_accuracy = 0.9

        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "temperature"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.05, temperature=1.8))
            strategy._fit_isotonic    = MagicMock()

            strategy._run_calibration(server_round=110)  # bem depois da melhor rodada (42)

        strategy.global_model.load_state_dict.assert_called_once_with(fake_best, strict=False)
        save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert save_kwargs["round_num"] == 42
        assert save_kwargs["accuracy"] == pytest.approx(0.9)

    def test_no_best_state_dict_skips_reload_but_still_uses_best_round(self):
        """Caso degenerado (nunca melhorou): não troca os pesos, mas ainda usa
        best_round/best_accuracy pra registrar corretamente."""
        strategy = _make_strategy()
        strategy._best_state_dict = None
        strategy._best_round = 5
        strategy._best_accuracy = 0.6

        with patch("mosaicfl.core.config.FED_CFG") as fed_cfg, \
             patch("mosaicfl.core.evaluation.evaluate", return_value=_fake_report(0.1)):
            fed_cfg.calibration_method = "temperature"
            strategy._fit_temperature = MagicMock(return_value=_fit_result("temperature", 0.05))
            strategy._fit_isotonic    = MagicMock()

            strategy._run_calibration(server_round=8)

        strategy.global_model.load_state_dict.assert_not_called()
        save_kwargs = strategy._checkpoint_store.save.call_args.kwargs
        assert save_kwargs["round_num"] == 5
