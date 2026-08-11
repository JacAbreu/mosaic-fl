"""base.py — Interface abstrata para persistência de checkpoints federados."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional


class CheckpointStore(ABC):
    """Interface para persistência de checkpoints federados."""

    @abstractmethod
    def register_training(
        self,
        algorithm: str = "FedAvg",
        log_file: str = "",
        n_rounds_max: int = 120,
        checkpoint_criterion: str = "f1_macro",
        partition_mode: str = "natural",
        run_classification: str = "ajuste",
        local_only_hospital: Optional[str] = None,
        early_stop_enabled: bool = False,
    ) -> int:
        """Registra um novo treinamento antes do loop FL. Retorna training_id.

        run_classification: "ajuste" (default — tuning/debugging/validação, NÃO
        citar como resultado final) ou "treinamento_real" (resultado formal para
        o texto de defesa). Precisa ser declarado explicitamente via
        FL_RUN_CLASSIFICATION — nunca fica ambíguo/dependente de doc externo.

        partition_mode: "natural" (hospital real = cliente) ou "iid_simulado"
        (pool embaralhado — Experimento 3 / fase 5, contraste non-IID vs. IID).

        local_only_hospital: None (treino federado normal) ou "BPSP"/"HSL"
        (Caminho B rodado com um único hospital conectado, min-clients=1 —
        baseline local pra comparar contra o federado na mesma rede real,
        migration 030).

        early_stop_enabled: valor de FED_CFG.early_stop no momento do registro
        (achado 2026-08-08, migration 036) — sem isso, n_rounds_done < n_rounds_max
        é ambíguo (parou por convergência real ou por outro motivo? já se
        confundiu nesta mesma fase, training_id 3/4)."""

    @abstractmethod
    def complete_training(
        self,
        training_id: int,
        n_rounds_done: int,
        best_round: int,
        best_accuracy: float,
        converged: bool,
        total_duration_s: float = 0.0,
        peak_ram_mb: float = 0.0,
        avg_cpu_pct: float = 0.0,
        gpu_avg_power_w: Optional[float] = None,
        gpu_peak_power_w: Optional[float] = None,
        gpu_energy_wh: Optional[float] = None,
        convergence_round: Optional[int] = None,
    ) -> None:
        """Atualiza fl_trainings com resultado final ao término do loop FL.

        gpu_*: None quando não há GPU NVIDIA disponível (treino CPU-only) — não é erro.

        convergence_round (migration 035): rodada real (server_round) em que a
        convergência foi detectada pela primeira vez, ou None se nunca convergiu.
        NÃO é o mesmo que ConvergenceTracker.converged_round (índice interno da
        janela deslizante — ver docstring da migration 035). Útil pra comparar
        contra best_round em /fl-training-results: quando divergem bastante,
        "convergência" (métrica estável) não coincidiu com "melhor qualidade"
        (achado real no treino 85, DP-uniforme: best_round=4, convergence_round=74)."""

    @abstractmethod
    def update_evaluation_metrics(
        self,
        training_id: int,
        macro_auc: Optional[float] = None,
        macro_f1: Optional[float] = None,
        ece: Optional[float] = None,
        ece_pre: Optional[float] = None,
        dp_noise_multiplier: Optional[float] = None,
        dp_max_grad_norm: Optional[float] = None,
        dp_epsilon_simple: Optional[float] = None,
        dp_epsilon_rdp: Optional[float] = None,
        dp_noise_strategy: Optional[str] = None,
        dp_noise_group_multipliers_json: Optional[str] = None,
        rag_precision_at_k: Optional[float] = None,
        rag_k: Optional[int] = None,
        calibration_method_requested: Optional[str] = None,
    ) -> None:
        """Grava em fl_trainings o AUC-ROC/F1/ECE pós-calibração (+ ECE pré-calibração,
        ece_pre) e, quando DP-FedAvg está habilitado, os parâmetros e o ε acumulado
        (composição simples e RDP). Calculados após complete_training() (a avaliação
        final só roda depois do melhor checkpoint ser restaurado). Chamado uma vez
        por treinamento, ao final da calibração.

        dp_noise_strategy/dp_noise_group_multipliers_json (migration 029): qual
        estratégia de ruído (mosaicfl.core.dp_noise) foi usada e os multiplicadores
        efetivos por grupo — "uniform" sempre resulta em {"all": dp_noise_multiplier};
        "layer_group" varia por camada. NULL quando DP está desligado.

        rag_precision_at_k/rag_k (migration 031): qualidade da recuperação do RAG —
        fração dos k casos recuperados que têm o mesmo desfecho da consulta,
        agregada entre clientes (mosaicfl.core.rag.precision.eval_precision_at_k,
        rodado localmente por hospital, nunca centraliza amostra). NULL quando
        nenhum cliente recebeu rag_patterns_json ainda (primeira rodada de
        avaliação de cada treino).

        calibration_method_requested (migration 034): valor de FED_CFG.calibration_method
        visto pelo ServerApp neste treino — não o método que cada cliente escolheu sob
        "auto" (isso já está em fl_round_history.calibration_per_client_json), mas o
        que foi PEDIDO. Existe pra distinguir "auto que escolheu temperature" de
        "temperature forçado direto" sem depender de log de cliente (achado 2026-07-29:
        FL_CALIBRATION_METHOD só tem efeito quando definida ao subir o SuperLink, não
        no comando server-app — ver Makefile, alvo "superlink")."""

    @abstractmethod
    def save(
        self,
        round_num: int,
        state_dict: OrderedDict,
        vocab: Dict[str, int],
        accuracy: float = 0.0,
        loss: float = 0.0,
        temperature: float = 1.0,
        training_id: Optional[int] = None,
        evaluation_json: Optional[Dict] = None,
        calibration_method: str = "temperature",
        isotonic_calibrators: Optional[List] = None,
        isotonic_num_classes: int = 0,
    ) -> None:
        """UPSERT do checkpoint: 1 linha por training_id (substitui quando Acc melhora).

        calibration_method: "temperature" | "isotonic" — qual calibrador está ativo neste
        checkpoint (ver FED_CFG.calibration_method, mosaicfl.core.config). Quando "isotonic",
        isotonic_calibrators/isotonic_num_classes devem ser passados (ver
        IsotonicCalibrator.calibrators); quando "temperature", o campo `temperature` já cobre
        a calibração e isotonic_calibrators deve ficar None."""

    @abstractmethod
    def save_round_history(
        self,
        training_id: int,
        rounds: list,
        accuracies: list,
        losses: list,
        tau_effs: Optional[list] = None,
        f1_macros: Optional[list] = None,
        per_class_f1s: Optional[list] = None,
        round_durations: Optional[list] = None,
        resource_per_client_jsons: Optional[list] = None,
        calibration_per_client_jsons: Optional[list] = None,
        per_client_f1_jsons: Optional[list] = None,
    ) -> None:
        """Persiste accuracy, loss, f1_macro, τ_eff, per_class_f1, round_duration_s e o
        detalhamento por cliente (recurso/calibração/per_class_f1, todos já serializados
        como string JSON por elemento) por rodada. tau_effs é None por elemento quando o
        algoritmo é FedAvg. UPSERT por (training_id, round) — chamar a cada rodada é seguro."""

    @abstractmethod
    def mark_active_model(self, training_id: int) -> None:
        """Marca training_id como o modelo ativo (is_active_model=TRUE) — desmarca
        qualquer outro. Substitui experiments/last_federated_training_id.txt: a API
        de inferência lê get_active_training_id() em vez de um arquivo local, que
        não é compartilhado entre processos/máquinas físicas diferentes."""

    @abstractmethod
    def get_active_training_id(self) -> Optional[int]:
        """training_id marcado como ativo (mark_active_model), ou None se nenhum."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna {'model_state': OrderedDict, 'vocab': dict} do checkpoint mais recente, ou None."""

    @abstractmethod
    def load_best(self, training_id: Optional[int] = None) -> Optional[Dict]:
        """Retorna o checkpoint com maior acurácia do treinamento indicado.
        Se training_id=None, usa o comportamento legado (melhor global — evitar)."""
