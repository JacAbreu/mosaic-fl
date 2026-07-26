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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import flwr as fl
import torch

from mosaicfl.core.config import FED_CFG, MODEL_CFG, RUNTIME_CFG
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
    _best_f1_macro      = 0.0   # fallback para __new__ em testes
    _best_macro_auc     = None  # fallback para __new__ em testes
    _best_criterion_value = 0.0  # fallback para __new__ em testes
    _training_completed = False  # fallback para __new__ em testes
    _last_round_resources_json = None  # fallback para __new__ em testes

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

        # Acumula custo computacional entre rodadas (mesmo padrão de manual_loop.py,
        # Caminho A) — só recebe valores quando os clientes enviam resource_* em
        # aggregate_fit() (ver FED_CFG.collect_resource_metrics/aggregate_resource_metrics
        # em federated.py). Energia é somada (custo real acumulado do treino inteiro);
        # potência é amostrada por rodada e usada pra média/pico ao final.
        self._gpu_energy_wh_total: float = 0.0
        self._gpu_power_samples: List[float] = []
        # peak_ram_mb/avg_cpu_pct de fl_trainings ficavam sempre em 0.0 no Caminho B —
        # a captura por cliente/rodada (resource_per_client_json) já existia, mas nunca
        # tinha sido acumulada em nível de treino nem passada pro complete_training()
        # (só a energia da GPU tinha esse acúmulo). Achado 2026-07-26, mesma sessão que
        # consertou o mesmo tipo de lacuna pra energia.
        self._peak_ram_mb: float = 0.0
        self._cpu_pct_samples: List[float] = []
        self._train_start_time: float = time.time()
        # Detalhamento por cliente da rodada atual (energia/CPU/RAM antes da
        # agregação) — capturado em aggregate_fit(), consumido em aggregate_evaluate()
        # pra persistir junto com o resto do histórico da rodada (achado 2026-07-26:
        # antes só existia em log, nunca em banco — ver migration 025).
        self._last_round_resources_json: Optional[str] = None

        self._training_id = training_id
        self._num_rounds = num_rounds
        self._run_id = run_id
        self._best_round = 0
        self._best_accuracy = 0.0
        self._best_f1_macro: float = 0.0
        self._best_macro_auc: Optional[float] = None
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

    def initialize_parameters(self, client_manager):
        """Roda uma única vez, antes da Rodada 1 — nunca de novo durante o treino.

        Ponto de extensão oficial do Flower pra isso: se retornar um valor não-None,
        o fallback nativo ("pedir parâmetros de um cliente aleatório") nunca dispara
        (ver flwr/server/server.py::_get_initial_parameters). Aproveitado aqui pra
        rodar a descoberta de vocabulário bidirecional ANTES de delegar pro
        comportamento padrão — vocabulário federado (self.vocab) fica decidido e
        congelado antes de qualquer rodada de treino de verdade começar, exatamente
        como a autora pediu (2026-07-26): "o vocabulário inserido a ser considerado
        pode ser o que foi inserido até o início do treinamento".
        """
        self._discover_and_curate_vocab(client_manager)
        return super().initialize_parameters(client_manager)

    def _discover_and_curate_vocab(self, client_manager) -> None:
        """Pede a cada cliente conectado os analitos locais que o vocab atual não
        cobre (FedProxClient.evaluate(discover_vocab_only=True) — reaproveita o RPC
        evaluate() já existente, não inventa mensagem nova), mescla as contribuições
        (soma n_records, conta n_hospitals = nº de clientes que reportaram o mesmo
        analito), aplica o mesmo critério de scripts/discover_analyte_catalog_gaps.py
        e atualiza self.vocab — que a partir daqui é a única fonte usada em
        configure_fit()/configure_evaluate() (ver fit_config_mixin.py) pro resto do
        treino inteiro.

        Nunca propaga exceção: descoberta de vocabulário é enriquecimento, uma falha
        aqui (cliente fora do ar, timeout, erro de SQL) não pode travar o treino —
        self.vocab simplesmente permanece como estava.
        """
        try:
            from flwr.common import EvaluateIns, ndarrays_to_parameters

            min_clients = getattr(self, "min_available_clients", 1) or 1
            timeout = float(os.getenv("FL_VOCAB_DISCOVERY_TIMEOUT", "120"))
            if not client_manager.wait_for(min_clients, timeout=int(timeout)):
                logger.warning(
                    "vocab_discovery_wait_timeout",
                    extra={"min_clients": min_clients, "timeout_s": timeout},
                )
                return

            clients = list(client_manager.all().values())
            if not clients:
                return

            ins = EvaluateIns(
                parameters=ndarrays_to_parameters([]),
                config={"discover_vocab_only": True, "vocab_json": json.dumps(self.vocab)},
            )

            per_client_candidates: List[list] = []
            with ThreadPoolExecutor(max_workers=len(clients)) as executor:
                futures = {
                    executor.submit(c.evaluate, ins, timeout, 0): c for c in clients
                }
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        cands = json.loads(res.metrics.get("vocab_candidates_json", "[]"))
                        per_client_candidates.append(cands)
                    except Exception as e:
                        logger.warning("vocab_discovery_client_error error=%s", e)

            merged: Dict[str, dict] = {}
            for cands in per_client_candidates:
                for c in cands:
                    entry = merged.setdefault(
                        c["analyte"], {"analyte": c["analyte"], "n_records": 0,
                                       "n_hospitals": 0, "has_real_ref": False}
                    )
                    entry["n_records"] += c.get("n_records", 0)
                    entry["n_hospitals"] += 1
                    entry["has_real_ref"] = entry["has_real_ref"] or bool(c.get("has_real_ref"))

            if not merged:
                logger.info("vocab_discovery_no_candidates")
                return

            from scripts.discover_analyte_catalog_gaps import select_insertable, insert_candidates
            from scripts.build_standard_vocab import build_standard_vocab

            selected = select_insertable(list(merged.values()))
            if not selected:
                logger.info(
                    "vocab_discovery_candidates_rejected",
                    extra={"n_candidates": len(merged)},
                )
                return

            import sqlalchemy as sa
            engine = sa.create_engine(RUNTIME_CFG.db_url)
            with engine.connect() as conn:
                n_inserted = insert_candidates(conn, selected)

            new_vocab = build_standard_vocab(RUNTIME_CFG.db_url)
            if len(new_vocab) > MODEL_CFG.vocab_size:
                logger.error(
                    "vocab_discovery_overflow tokens=%d vocab_size=%d — mantendo vocab anterior",
                    len(new_vocab), MODEL_CFG.vocab_size,
                )
                return

            old_size = len(self.vocab)
            self.vocab = new_vocab
            logger.info(
                "vocab_discovery_curated",
                extra={
                    "analitos_inseridos": n_inserted,
                    "vocab_antes": old_size,
                    "vocab_depois": len(self.vocab),
                },
            )
        except Exception as e:
            logger.warning("vocab_discovery_error error=%s", e)

    def aggregate_fit(self, server_round, results, failures):
        """Agrega pesos e salva checkpoint. Cancela watchdog do round."""
        self._cancel_round_watchdog()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_metrics:
            round_gpu_energy = aggregated_metrics.get("resource_round_gpu_energy_wh")
            if round_gpu_energy:
                self._gpu_energy_wh_total += round_gpu_energy
            round_gpu_avg_power = aggregated_metrics.get("resource_round_gpu_avg_power_w")
            if round_gpu_avg_power is not None:
                self._gpu_power_samples.append(round_gpu_avg_power)
            round_peak_ram = aggregated_metrics.get("resource_round_peak_ram_mb")
            if round_peak_ram is not None:
                self._peak_ram_mb = max(self._peak_ram_mb, round_peak_ram)
            round_avg_cpu = aggregated_metrics.get("resource_round_avg_cpu_pct")
            if round_avg_cpu is not None:
                self._cpu_pct_samples.append(round_avg_cpu)
            if "resource_per_client_json" in aggregated_metrics:
                self._last_round_resources_json = aggregated_metrics["resource_per_client_json"]
                logger.info(
                    "round_resources",
                    extra={
                        "round": server_round,
                        "gpu_energy_wh": round_gpu_energy,
                        "gpu_avg_power_w": round_gpu_avg_power,
                        "per_client": aggregated_metrics["resource_per_client_json"],
                    },
                )
            else:
                self._last_round_resources_json = None

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
        macro_auc = aggregated_metrics.get("macro_auc") if aggregated_metrics else None
        per_class_f1_json = aggregated_metrics.get("per_class_f1_json") if aggregated_metrics else None
        per_class_f1 = json.loads(per_class_f1_json) if per_class_f1_json else None
        rag_patterns_json = aggregated_metrics.get("rag_patterns_json") if aggregated_metrics else None
        calibration_method = aggregated_metrics.get("calibration_method") if aggregated_metrics else None

        # Calibração por cliente ANTES da agregação — só o T/isotônico federado final
        # (pós-agregação) ia pro checkpoint; o valor individual de cada hospital só
        # existia no log do próprio cliente (local_calibration_fit). Extraído de
        # "results" (raw, per-cliente) porque aggregated_metrics já é o resultado
        # combinado — achado 2026-07-26, ver migration 025.
        per_client_calibration = []
        for _, evaluate_res in results:
            m = getattr(evaluate_res, "metrics", None) or {}
            if "calibration_method" in m:
                entry = {"client_id": m.get("client_id"), "method": m["calibration_method"]}
                if m["calibration_method"] == "temperature" and "temperature" in m:
                    entry["temperature"] = m["temperature"]
                per_client_calibration.append(entry)
        calibration_per_client_json = json.dumps(per_client_calibration) if per_client_calibration else None

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

        # Persistência incremental — a cada rodada, não só no fim do treino. Achado
        # 2026-07-26: antes, fl_round_history só recebia dados via save_round_history()
        # em lote, uma única vez, ao concluir o treino — se o processo caísse no meio,
        # o histórico rodada-a-rodada só sobrevivia em logs/round_N_metrics.json
        # (arquivo), nunca no banco. Nunca propaga exceção — persistência de histórico
        # é enriquecimento, não deve travar o treino.
        if self._checkpoint_store is not None and self._training_id is not None:
            try:
                self._checkpoint_store.save_round_history(
                    training_id=self._training_id,
                    rounds=[server_round],
                    accuracies=[accuracy],
                    losses=[aggregated_loss],
                    f1_macros=[f1_macro],
                    per_class_f1s=[per_class_f1 or []],
                    resource_per_client_jsons=[self._last_round_resources_json],
                    calibration_per_client_jsons=[calibration_per_client_json],
                )
            except Exception as e:
                logger.warning(
                    "round_history_incremental_save_error",
                    extra={"training_id": self._training_id, "round": server_round, "error": str(e)},
                )

        # Critério de melhor rodada segue FED_CFG.checkpoint_criterion (já declarado
        # em fl_trainings.checkpoint_criterion, ver register_training() em
        # superlink.py) — mesma lógica de manual_loop.py (Caminho A): best_accuracy
        # sempre guarda a accuracy (não o F1) da rodada escolhida, mesmo quando o
        # critério de escolha é f1_macro — só o critério de SELEÇÃO muda.
        criterion_value = f1_macro if FED_CFG.checkpoint_criterion == "f1_macro" else accuracy
        if criterion_value > self._best_criterion_value:
            self._best_criterion_value = criterion_value
            self._best_accuracy = accuracy
            self._best_f1_macro = f1_macro
            self._best_macro_auc = macro_auc
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
            gpu_avg_power_w  = sum(self._gpu_power_samples) / len(self._gpu_power_samples) if self._gpu_power_samples else None
            gpu_peak_power_w = max(self._gpu_power_samples) if self._gpu_power_samples else None
            gpu_energy_wh    = self._gpu_energy_wh_total if self._gpu_energy_wh_total > 0 else None
            total_duration_s = time.time() - self._train_start_time
            avg_cpu_pct      = sum(self._cpu_pct_samples) / len(self._cpu_pct_samples) if self._cpu_pct_samples else 0.0
            self._checkpoint_store.complete_training(
                training_id=self._training_id,
                n_rounds_done=server_round,
                best_round=self._best_round,
                best_accuracy=self._best_accuracy,
                converged=converged,
                total_duration_s=total_duration_s,
                peak_ram_mb=self._peak_ram_mb,
                avg_cpu_pct=avg_cpu_pct,
                gpu_avg_power_w=gpu_avg_power_w,
                gpu_peak_power_w=gpu_peak_power_w,
                gpu_energy_wh=gpu_energy_wh,
            )
            logger.info(
                "training_completed",
                extra={
                    "training_id": self._training_id,
                    "n_rounds_done": server_round,
                    "best_round": self._best_round,
                    "best_accuracy": self._best_accuracy,
                    "converged": converged,
                    "total_duration_s": total_duration_s,
                    "peak_ram_mb": self._peak_ram_mb,
                    "avg_cpu_pct": avg_cpu_pct,
                    "gpu_avg_power_w": gpu_avg_power_w,
                    "gpu_peak_power_w": gpu_peak_power_w,
                    "gpu_energy_wh": gpu_energy_wh,
                },
            )
            self._save_federated_training_id_marker()
            try:
                self._checkpoint_store.update_evaluation_metrics(
                    training_id=self._training_id,
                    macro_f1=self._best_f1_macro,
                    macro_auc=self._best_macro_auc,
                )
                logger.info(
                    "evaluation_metrics_updated",
                    extra={
                        "training_id": self._training_id,
                        "macro_f1": self._best_f1_macro,
                        "macro_auc": self._best_macro_auc,
                    },
                )
            except Exception as e:
                logger.warning(
                    "evaluation_metrics_update_error",
                    extra={"training_id": self._training_id, "error": str(e)},
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

        if calibration_method:
            self._persist_federated_calibration(server_round, calibration_method, aggregated_metrics)

        return aggregated_loss, aggregated_metrics

    def _save_federated_training_id_marker(self) -> None:
        """Marca self._training_id como o modelo ativo em fl_trainings.is_active_model
        (CheckpointStore.mark_active_model) — a API de inferência lê essa coluna em vez
        de um arquivo local (experiments/last_federated_training_id.txt, achado
        2026-07-25/26: arquivo não é fonte de verdade compartilhada entre processos/
        máquinas físicas diferentes, ver migration 024). Nunca propaga exceção: é um
        atalho de conveniência, não uma dependência crítica do treino (sem isso, a API
        só cai no fallback de maior accuracy global)."""
        try:
            self._checkpoint_store.mark_active_model(self._training_id)
        except Exception as e:
            logger.warning(
                "active_model_mark_error",
                extra={"training_id": self._training_id, "error": str(e)},
            )

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

    def _persist_federated_calibration(
        self, server_round: int, calibration_method: str, aggregated_metrics: Dict,
    ) -> None:
        """Persiste o calibrador federado agregado (ver aggregate_calibration em
        federated.py) no checkpoint — cada cliente ajusta localmente e envia só
        estatísticas comprimidas/agregadas (nunca dado bruto por amostra); esta função
        só combina o que já chegou agregado do lado do cliente, num único calibrador
        pronto pra InferenceEngine.calibrate()/calibrate_probs() carregar depois.
        Nunca propaga exceção: calibração é enriquecimento pós-convergência, uma
        falha aqui não deve invalidar o checkpoint do treino em si."""
        if self._checkpoint_store is None:
            return
        try:
            isotonic_calibrators = None
            isotonic_num_classes = 0
            temperature = 1.0

            if calibration_method == "isotonic":
                pooled_json = aggregated_metrics.get("isotonic_pooled_thresholds_json")
                num_classes = aggregated_metrics.get("isotonic_num_classes", 0)
                if not pooled_json or not num_classes:
                    logger.warning("federated_calibration_incomplete method=isotonic round=%d", server_round)
                    return
                from mosaicfl.core.calibration import IsotonicCalibrator
                iso = IsotonicCalibrator.from_pooled_thresholds(json.loads(pooled_json), num_classes)
                isotonic_calibrators = iso.calibrators
                isotonic_num_classes = num_classes
            elif calibration_method == "temperature":
                temperature = float(aggregated_metrics.get("temperature", 1.0))
            else:
                logger.warning("federated_calibration_unknown_method method=%s", calibration_method)
                return

            last_acc = self._last_round_metrics.get("accuracy", 0.0)
            last_loss = self._last_round_metrics.get("loss", 0.0)
            self._checkpoint_store.save(
                round_num=server_round,
                state_dict=self.global_model.state_dict(),
                vocab=self.vocab,
                training_id=self._training_id,
                accuracy=last_acc,
                loss=last_loss,
                calibration_method=calibration_method,
                temperature=temperature,
                isotonic_calibrators=isotonic_calibrators,
                isotonic_num_classes=isotonic_num_classes,
            )
            logger.info(
                "federated_calibration_persisted",
                extra={"round": server_round, "method": calibration_method, "temperature": temperature},
            )
        except Exception as e:
            logger.warning("federated_calibration_persist_error round=%d error=%s", server_round, e)

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
