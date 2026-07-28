"""calibration_mixin.py — Calibração pós-convergência (temperature scaling e/ou isotônica) + avaliação clínica.

Método controlado por FED_CFG.calibration_method ("temperature" | "isotonic" | "auto" —
ver mosaicfl.core.config). Nenhum dos três é federado sob DP ainda (ver
docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 9) — a calibração
roda sobre o modelo global já agregado, usando o holdout local do servidor (_test_loader).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class _CalibrationMixin:
    """Requer os atributos definidos em ProductionFedProxStrategy.__init__ (global_model,
    _test_loader, _checkpoint_store, vocab, _last_round_metrics, _best_round,
    _best_accuracy, _best_state_dict) e o método _save_evaluation_report (de core.py,
    via herança na classe final)."""

    def _fit_temperature(self, device: str) -> Optional[dict]:
        """Ajusta TemperatureScaler e avalia. Retorna None se falhar em qualquer etapa."""
        from mosaicfl.core.calibration import TemperatureScaler
        from mosaicfl.core.config import MODEL_CFG
        from mosaicfl.core.evaluation import evaluate

        try:
            scaler = TemperatureScaler()
            scaler.fit(self.global_model, self._test_loader, device=device)
        except Exception as exc:
            logger.warning("temperature_calibration_error %s", exc)
            return None

        try:
            report_cal = evaluate(
                self.global_model,
                self._test_loader,
                class_labels=MODEL_CFG.class_labels,
                device=device,
                temperature=scaler.T,
            )
        except Exception as exc:
            logger.warning("temperature_evaluation_error %s", exc)
            return None

        return {
            "method":                "temperature",
            "temperature":           scaler.T,
            "isotonic_calibrators":  None,
            "isotonic_num_classes":  0,
            "report_cal":            report_cal,
        }

    def _fit_isotonic(self, device: str) -> Optional[dict]:
        """Ajusta IsotonicCalibrator (OvR) e avalia. Retorna None se falhar em qualquer etapa."""
        from mosaicfl.core.calibration import IsotonicCalibrator
        from mosaicfl.core.config import MODEL_CFG
        from mosaicfl.core.evaluation import collect_logits, report_from_probs

        try:
            iso = IsotonicCalibrator()
            iso.fit(self.global_model, self._test_loader, device=device, num_classes=MODEL_CFG.num_classes)
        except Exception as exc:
            logger.warning("isotonic_calibration_error %s", exc)
            return None

        try:
            raw_probs, all_labels = collect_logits(self.global_model, self._test_loader, device, temperature=1.0)
            cal_probs  = iso.calibrate_probs(raw_probs)
            report_cal = report_from_probs(cal_probs, all_labels, MODEL_CFG.class_labels, temperature=1.0)
        except Exception as exc:
            logger.warning("isotonic_evaluation_error %s", exc)
            return None

        return {
            "method":                "isotonic",
            "temperature":           1.0,
            "isotonic_calibrators":  iso.calibrators,
            "isotonic_num_classes":  MODEL_CFG.num_classes,
            "report_cal":            report_cal,
        }

    def _run_calibration(self, server_round: int) -> None:
        """Executa calibração pós-convergência + avaliação clínica.

        Com FL_DB_URL configurado (dados reais):
          1. Avalia o modelo antes da calibração (ECE com T=1.0) — linha de base, comum aos 3 modos
          2. Ajusta o(s) calibrador(es) conforme FED_CFG.calibration_method:
             "temperature" — TemperatureScaler via LBFGS (padrão)
             "isotonic"    — IsotonicCalibrator OvR (PAV por classe)
             "auto"        — ajusta os dois e mantém o de menor ECE no conjunto de calibração
          3. Avalia novamente com o calibrador escolhido
          4. Salva relatório em logs/evaluation_round_N.json
          5. Re-persiste checkpoint com o calibrador escolhido

        Sem FL_DB_URL (simulação com dados sintéticos):
          test_loader é None — calibração e avaliação são puladas com aviso.
        """
        if self._test_loader is None:
            logger.warning(
                "calibration_skipped — test_loader ausente; "
                "configure FL_DB_URL e FL_TEST_HOLDOUT_FRACTION para habilitar"
            )
            return

        from mosaicfl.core.config import FED_CFG, MODEL_CFG, RUNTIME_CFG
        from mosaicfl.core.evaluation import evaluate

        # Achado 2026-07-28 (mesma classe do bug já corrigido em aggregate_fit()/
        # _persist_federated_calibration, ver docs/Linha_do_Tempo_MOSAIC-FL.md):
        # calibração precisa avaliar/ajustar sobre os pesos da MELHOR rodada, não
        # os da rodada em que foi disparada (server_round pode ser bem depois do
        # best_round — ex.: converged=True fica permanente, então este método pode
        # rodar de novo em rodadas seguintes). Carrega self._best_state_dict em
        # self.global_model ANTES de qualquer avaliação — sem isso, o relatório e
        # os calibradores ficariam ajustados sobre um modelo diferente do que
        # termina salvo/servido. Seguro mutar self.global_model aqui: a próxima
        # rodada sempre recarrega os pesos agregados de verdade via
        # aggregate_fit()/_load_global_weights(), então essa troca é só temporária.
        best_round = self._best_round or server_round
        if self._best_state_dict is not None:
            self.global_model.load_state_dict(self._best_state_dict, strict=False)

        device = str(RUNTIME_CFG.device)
        method = FED_CFG.calibration_method

        # 1. Avaliação ANTES da calibração (T=1.0) — linha de base, comum aos 3 modos
        try:
            report_raw = evaluate(
                self.global_model,
                self._test_loader,
                class_labels=MODEL_CFG.class_labels,
                device=device,
                temperature=1.0,
            )
            logger.info(
                "evaluation_pre_calibration",
                extra={
                    "round":      best_round,
                    "accuracy":   report_raw.accuracy,
                    "macro_f1":   report_raw.macro_f1,
                    "macro_auc":  report_raw.macro_auc,
                    "ece":        report_raw.calibration.ece,
                    "mce":        report_raw.calibration.mce,
                    "n_samples":  report_raw.n_samples,
                },
            )
        except Exception as exc:
            logger.warning("evaluation_pre_calibration_error %s", exc)
            report_raw = None

        # 2. Ajusta o(s) calibrador(es) conforme o método configurado
        if method == "isotonic":
            result = self._fit_isotonic(device)
        elif method == "auto":
            temp_result = self._fit_temperature(device)
            iso_result  = self._fit_isotonic(device)
            candidates  = [r for r in (temp_result, iso_result) if r is not None]
            if not candidates:
                result = None
            else:
                result = min(candidates, key=lambda r: r["report_cal"].calibration.ece)
                logger.info(
                    "calibration_auto_selected",
                    extra={
                        "round":  best_round,
                        "method": result["method"],
                        "ece":    result["report_cal"].calibration.ece,
                    },
                )
        else:
            if method != "temperature":
                logger.warning("calibration_method_unknown method=%s — usando temperature", method)
            result = self._fit_temperature(device)

        if result is None:
            logger.warning("calibration_failed method=%s — checkpoint mantido sem recalibração", method)
            return

        report_cal = result["report_cal"]
        logger.info(
            "calibration_complete",
            extra={
                "round":     best_round,
                "method":    result["method"],
                "T":         round(result["temperature"], 4),
                "ece":       report_cal.calibration.ece,
                "mce":       report_cal.calibration.mce,
                "macro_auc": report_cal.macro_auc,
                "macro_f1":  report_cal.macro_f1,
            },
        )

        # 3. Persiste relatório em JSON para rastreabilidade clínica
        self._save_evaluation_report(
            best_round, result["temperature"], report_raw, report_cal,
            calibration_method=result["method"],
        )

        # 4. Re-persiste checkpoint com o calibrador escolhido — pesos da MELHOR
        # rodada (já carregados em self.global_model no início deste método),
        # não os "últimos" (achado 2026-07-28).
        if self._checkpoint_store is not None:
            self._checkpoint_store.save(
                round_num=best_round,
                state_dict=self.global_model.state_dict(),
                vocab=self.vocab,
                accuracy=self._best_accuracy,
                loss=0.0,  # loss da melhor rodada não é rastreado separadamente, só accuracy/f1_macro
                temperature=result["temperature"],
                calibration_method=result["method"],
                isotonic_calibrators=result["isotonic_calibrators"],
                isotonic_num_classes=result["isotonic_num_classes"],
            )
