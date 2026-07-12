"""
core.py — ProductionFedProxStrategy: __init__, checkpoint em aggregate_fit, métricas em
aggregate_evaluate e persistência do relatório de avaliação.

CHECKPOINT_DIR e LOG_DIR ficam aqui (não em constants.py) porque os testes fazem
patch direto em "infrastructure.mosaicfl_server.strategy.core.CHECKPOINT_DIR" /
".core.LOG_DIR" — os métodos que leem essas constantes precisam estar neste módulo
para o patch ser efetivo (mock intercepta onde é lido, não onde é definido).
"""
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import flwr as fl
import torch

from mosaicfl.core.config import FED_CFG
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.federated import weighted_average_accuracy, weighted_average_loss

from ..config_loader import ConfigLoader, get_config_loader
from ..state_store import TrainingState, TrainingStateStore
from infrastructure.shared.checkpoint_store import CheckpointStore

from .calibration_mixin import _CalibrationMixin
from .fit_config_mixin import _FitConfigMixin
from .watchdog_mixin import _WatchdogMixin

CHECKPOINT_DIR = Path(os.getenv("FL_CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))

logger = logging.getLogger(__name__)


class ProductionFedProxStrategy(
    _FitConfigMixin,
    _WatchdogMixin,
    _CalibrationMixin,
    fl.server.strategy.FedProx,
):
    """
    FedProx adaptado para produção:
      - Checkpoint do modelo global a cada rodada
      - Exporta métricas para JSON (consumidas pelo scheduler)
      - Rastreia convergência
      - Lê config de runtime do PostgreSQL (ou arquivo) antes de cada round
      - Temperature scaling pós-convergência (quando test_loader disponível)
    """

    _test_loader      = None  # fallback para __new__ em testes
    _checkpoint_store = None  # fallback para __new__ em testes
    _training_id        = None  # fallback para __new__ em testes
    _num_rounds         = None  # fallback para __new__ em testes
    _run_id             = None  # fallback para __new__ em testes
    _best_round         = 0     # fallback para __new__ em testes
    _best_accuracy      = 0.0   # fallback para __new__ em testes
    _best_criterion_value = 0.0  # fallback para __new__ em testes
    _training_completed = False  # fallback para __new__ em testes

    def __init__(
        self,
        global_model: torch.nn.Module,
        vocab: Optional[Dict[str, int]] = None,
        config_loader: Optional[ConfigLoader] = None,
        on_round_start: Optional[Callable[[int, Dict], None]] = None,
        on_round_complete: Optional[Callable[[int, Dict], None]] = None,
        state_store: Optional[TrainingStateStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        round_timeout: int = 300,
        test_loader=None,
        training_id: Optional[int] = None,
        num_rounds: Optional[int] = None,
        run_id: Optional[int] = None,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("evaluate_metrics_aggregation_fn", weighted_average_accuracy)
        kwargs.setdefault("fit_metrics_aggregation_fn", weighted_average_loss)
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.vocab: Dict[str, int] = vocab or {}
        self.config_loader: ConfigLoader = config_loader or get_config_loader()
        self.on_round_start = on_round_start
        self.on_round_complete = on_round_complete
        self.tracker = ConvergenceTracker(
            threshold=FED_CFG.convergence_threshold,
            patience=FED_CFG.convergence_patience,
        )
        self.round_counter = 0
        self.should_stop = False

        self._state_store = state_store
        self._checkpoint_store = checkpoint_store
        self._round_timeout = round_timeout
        self._test_loader = test_loader
        self._round_timer: Optional[threading.Timer] = None
        self._current_state = TrainingState()
        self._last_round_metrics: Dict = {}

        # Acumula por rodada para save_round_history() no fim do treino (mesmo padrão
        # de manual_loop.py, Caminho A). tau_eff/round_duration_s não são calculados
        # aqui — FedProx não usa tau_eff (só FedNova) e não há medição de duração
        # por rodada implementada no Caminho B ainda.
        self._history_rounds: List[int] = []
        self._history_accuracies: List[float] = []
        self._history_losses: List[float] = []
        self._history_f1_macros: List[float] = []
        self._history_per_class_f1: List[list] = []
        self._best_criterion_value: float = 0.0

        self._training_id = training_id
        self._num_rounds = num_rounds
        self._run_id = run_id
        self._best_round = 0
        self._best_accuracy = 0.0
        self._training_completed = False

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if state_store is not None:
            loaded_state = state_store.load()
            if loaded_state.run_id == run_id and run_id is not None:
                self._restore_from_state(loaded_state)
            else:
                # run_id diferente (ou estado antigo, sem run_id gravado) — não é
                # retomada da mesma sessão após queda, é um run novo de verdade.
                # Restaurar o ConvergenceTracker aqui reaproveitaria o histórico
                # (e um eventual converged_round) de um run anterior sem relação
                # com este, podendo disparar "convergência" falsa desde a 1ª rodada.
                logger.info(
                    "state_not_restored_new_run",
                    extra={"stored_run_id": loaded_state.run_id, "current_run_id": run_id},
                )
                self._current_state.run_id = run_id

    def _save_state(self, server_round: int) -> None:
        """Persiste estado atual no TrainingStateStore."""
        if self._state_store is None:
            return
        self._current_state.run_id = self._run_id
        self._current_state.last_round = server_round
        self._current_state.convergence_history = list(self.tracker.history)
        self._current_state.converged_round = self.tracker.converged_round
        self._current_state.last_metrics = self._last_round_metrics
        self._current_state.last_checkpoint = str(CHECKPOINT_DIR / f"round_{server_round}.pt")
        self._state_store.save(self._current_state)

    def aggregate_fit(self, server_round, results, failures):
        """Agrega pesos e salva checkpoint. Cancela watchdog do round."""
        self._cancel_round_watchdog()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._load_global_weights(aggregated_parameters)
            if self._checkpoint_store is not None:
                last_acc = self._last_round_metrics.get("accuracy", 0.0)
                last_loss = self._last_round_metrics.get("loss", 0.0)
                self._checkpoint_store.save(
                    round_num=server_round,
                    state_dict=self.global_model.state_dict(),
                    vocab=self.vocab,
                    training_id=self._training_id,
                    accuracy=last_acc,
                    loss=last_loss,
                )
                logger.info(
                    "checkpoint_saved",
                    extra={
                        "round": server_round,
                        "store": type(self._checkpoint_store).__name__,
                        "vocab_size": len(self.vocab),
                    },
                )
            else:
                checkpoint_path = CHECKPOINT_DIR / f"round_{server_round}.pt"
                from ..runner import _save_checkpoint
                _save_checkpoint(
                    checkpoint_path,
                    {"model_state": self.global_model.state_dict(), "vocab": self.vocab},
                )
                logger.info(
                    "checkpoint_saved",
                    extra={"round": server_round, "path": str(checkpoint_path), "vocab_size": len(self.vocab)},
                )

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas e detecta convergência."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracy = aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0
        f1_macro = aggregated_metrics.get("f1_macro", 0.0) if aggregated_metrics else 0.0
        per_class_f1_json = aggregated_metrics.get("per_class_f1_json") if aggregated_metrics else None
        per_class_f1 = json.loads(per_class_f1_json) if per_class_f1_json else None
        rag_patterns_json = aggregated_metrics.get("rag_patterns_json") if aggregated_metrics else None

        self.tracker.check(accuracy)
        self.round_counter = server_round

        converged = self.tracker.converged_round is not None
        round_metrics = {
            "round": server_round,
            "loss": aggregated_loss,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "per_class_f1": per_class_f1,
            "timestamp": datetime.now().isoformat(),
            "converged": converged,
            "convergence_round": self.tracker.converged_round,
        }
        self._last_round_metrics = round_metrics

        self._history_rounds.append(server_round)
        self._history_accuracies.append(accuracy)
        self._history_losses.append(aggregated_loss)
        self._history_f1_macros.append(f1_macro)
        self._history_per_class_f1.append(per_class_f1 or [])

        # Critério de melhor rodada segue FED_CFG.checkpoint_criterion (já declarado
        # em fl_trainings.checkpoint_criterion, ver register_training() em
        # superlink.py) — mesma lógica de manual_loop.py (Caminho A): best_accuracy
        # sempre guarda a accuracy (não o F1) da rodada escolhida, mesmo quando o
        # critério de escolha é f1_macro — só o critério de SELEÇÃO muda.
        criterion_value = f1_macro if FED_CFG.checkpoint_criterion == "f1_macro" else accuracy
        if criterion_value > self._best_criterion_value:
            self._best_criterion_value = criterion_value
            self._best_accuracy = accuracy
            self._best_round = server_round

        metrics_file = LOG_DIR / f"round_{server_round}_metrics.json"
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(round_metrics, f, indent=2)
        except Exception as e:
            logger.warning("metrics_write_error", extra={"round": server_round, "error": str(e)})

        # Persiste estado após cada round — permite recovery exato no próximo restart
        self._current_state.status = "completed" if converged else "running"
        self._save_state(server_round)

        if self.on_round_complete is not None:
            try:
                self.on_round_complete(server_round, round_metrics)
            except Exception as e:
                logger.warning(
                    "round_complete_callback_error",
                    extra={"round": server_round, "error": str(e)},
                )

        if converged:
            self.should_stop = True
            logger.info(
                "convergence_detected",
                extra={"round": server_round, "convergence_round": self.tracker.converged_round},
            )
            self._run_calibration(server_round)

        is_last_round = self._num_rounds is not None and server_round >= self._num_rounds
        if (
            (converged or is_last_round)
            and not self._training_completed
            and self._checkpoint_store is not None
            and self._training_id is not None
        ):
            self._training_completed = True
            self._checkpoint_store.complete_training(
                training_id=self._training_id,
                n_rounds_done=server_round,
                best_round=self._best_round,
                best_accuracy=self._best_accuracy,
                converged=converged,
            )
            logger.info(
                "training_completed",
                extra={
                    "training_id": self._training_id,
                    "n_rounds_done": server_round,
                    "best_round": self._best_round,
                    "best_accuracy": self._best_accuracy,
                    "converged": converged,
                },
            )
            try:
                self._checkpoint_store.save_round_history(
                    training_id=self._training_id,
                    rounds=self._history_rounds,
                    accuracies=self._history_accuracies,
                    losses=self._history_losses,
                    f1_macros=self._history_f1_macros,
                    per_class_f1s=self._history_per_class_f1,
                )
                logger.info(
                    "round_history_saved",
                    extra={"training_id": self._training_id, "n_rounds": len(self._history_rounds)},
                )
            except Exception as e:
                logger.warning(
                    "round_history_save_error",
                    extra={"training_id": self._training_id, "error": str(e)},
                )

        if rag_patterns_json:
            self._build_rag_knowledge_base(rag_patterns_json)

        return aggregated_loss, aggregated_metrics

    def _build_rag_knowledge_base(self, patterns_json: str) -> None:
        """Constrói a base de conhecimento do RAG com os perfis prototípicos enviados
        pelos clientes (extraídos localmente, sem dado identificável de paciente —
        ver FedProxClient._extract_rag_patterns). Nunca propaga exceção: RAG é
        enriquecimento, uma falha aqui não deve travar o treino federado."""
        try:
            patterns = json.loads(patterns_json)
            from mosaicfl.core.rag import ClinicalRAG
            ClinicalRAG().build_knowledge_base(patterns)
            logger.info("rag_knowledge_base_built", extra={"n_patterns": len(patterns)})
        except Exception as e:
            logger.warning("rag_knowledge_base_build_error", extra={"error": str(e)})

    def _save_evaluation_report(
        self,
        server_round: int,
        temperature: float,
        report_raw,
        report_cal,
        calibration_method: str = "temperature",
    ) -> None:
        """Persiste relatório de avaliação clínica em JSON.

        temperature: T ajustado quando calibration_method="temperature"; 1.0 (sem
        significado numérico) quando calibration_method="isotonic" — o calibrador
        isotônico não tem um escalar equivalente, ver isotonic_calibrators no checkpoint."""
        import dataclasses

        def _to_dict(report):
            if report is None:
                return None
            d = dataclasses.asdict(report)
            # dataclasses.asdict converte tudo recursivamente, inclusive BinStats e ClassMetrics
            return d

        payload = {
            "round":              server_round,
            "calibration_method": calibration_method,
            "temperature":        round(temperature, 4),
            "pre_calibration":    _to_dict(report_raw),
            "post_calibration":   _to_dict(report_cal),
        }

        out_path = LOG_DIR / f"evaluation_round_{server_round}.json"
        try:
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("evaluation_report_saved path=%s", out_path)
        except Exception as exc:
            logger.warning("evaluation_report_save_error %s", exc)
