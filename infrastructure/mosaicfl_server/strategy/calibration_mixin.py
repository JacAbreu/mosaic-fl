"""calibration_mixin.py — Temperature scaling + avaliação clínica pós-convergência."""
import logging

logger = logging.getLogger(__name__)


class _CalibrationMixin:
    """Requer os atributos definidos em ProductionFedProxStrategy.__init__ (global_model,
    _test_loader, _checkpoint_store, vocab, _last_round_metrics) e o método
    _save_evaluation_report (de core.py, via herança na classe final)."""

    def _run_calibration(self, server_round: int) -> None:
        """Executa temperature scaling + avaliação clínica após convergência.

        Com FL_DB_URL configurado (dados reais):
          1. Avalia o modelo antes da calibração (ECE com T=1.0)
          2. Ajusta temperatura via LBFGS no holdout
          3. Avalia novamente (ECE com T=T_otimizado)
          4. Salva relatório em logs/evaluation_round_N.json
          5. Re-persiste checkpoint com T

        Sem FL_DB_URL (simulação com dados sintéticos):
          test_loader é None — calibração e avaliação são puladas com aviso.
        """
        if self._test_loader is None:
            logger.warning(
                "calibration_skipped — test_loader ausente; "
                "configure FL_DB_URL e FL_TEST_HOLDOUT_FRACTION para habilitar"
            )
            return

        from mosaicfl.core.calibration import TemperatureScaler
        from mosaicfl.core.config import MODEL_CFG, RUNTIME_CFG
        from mosaicfl.core.evaluation import evaluate, print_report

        device = str(RUNTIME_CFG.device)

        # 1. Avaliação ANTES da calibração (T=1.0) — linha de base
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
                    "round":      server_round,
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

        # 2. Calibração por temperature scaling
        try:
            scaler = TemperatureScaler()
            scaler.fit(self.global_model, self._test_loader, device=device)
            logger.info("calibration_complete", extra={"round": server_round, "T": round(scaler.T, 4)})
        except Exception as exc:
            logger.warning("calibration_error %s — T mantido em 1.0", exc)
            return

        # 3. Avaliação APÓS calibração (T=T_otimizado)
        try:
            report_cal = evaluate(
                self.global_model,
                self._test_loader,
                class_labels=MODEL_CFG.class_labels,
                device=device,
                temperature=scaler.T,
            )
            logger.info(
                "evaluation_post_calibration",
                extra={
                    "round":     server_round,
                    "T":         round(scaler.T, 4),
                    "ece":       report_cal.calibration.ece,
                    "mce":       report_cal.calibration.mce,
                    "macro_auc": report_cal.macro_auc,
                    "macro_f1":  report_cal.macro_f1,
                },
            )
        except Exception as exc:
            logger.warning("evaluation_post_calibration_error %s", exc)
            report_cal = None

        # 4. Persiste relatório em JSON para rastreabilidade clínica
        self._save_evaluation_report(server_round, scaler.T, report_raw, report_cal)

        # 5. Re-persiste checkpoint com T; load_latest() retorna o mais recente por id
        if self._checkpoint_store is not None:
            last_acc  = self._last_round_metrics.get("accuracy", 0.0)
            last_loss = self._last_round_metrics.get("loss", 0.0)
            self._checkpoint_store.save(
                round_num=server_round,
                state_dict=self.global_model.state_dict(),
                vocab=self.vocab,
                accuracy=last_acc,
                loss=last_loss,
                temperature=scaler.T,
            )
