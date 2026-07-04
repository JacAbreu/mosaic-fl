"""base.py — Interface abstrata para persistência de checkpoints federados."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional


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
    ) -> int:
        """Registra um novo treinamento antes do loop FL. Retorna training_id.

        run_classification: "ajuste" (default — tuning/debugging/validação, NÃO
        citar como resultado final) ou "treinamento_real" (resultado formal para
        o texto de defesa). Precisa ser declarado explicitamente via
        FL_RUN_CLASSIFICATION — nunca fica ambíguo/dependente de doc externo.

        partition_mode: "natural" (hospital real = cliente) ou "iid_simulado"
        (pool embaralhado — Experimento 3 / fase 5, contraste non-IID vs. IID)."""

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
    ) -> None:
        """Atualiza fl_trainings com resultado final ao término do loop FL.

        gpu_*: None quando não há GPU NVIDIA disponível (treino CPU-only) — não é erro."""

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
    ) -> None:
        """Grava em fl_trainings o AUC-ROC/F1/ECE pós-calibração (+ ECE pré-calibração,
        ece_pre) e, quando DP-FedAvg está habilitado, os parâmetros e o ε acumulado
        (composição simples e RDP). Calculados após complete_training() (a avaliação
        final só roda depois do melhor checkpoint ser restaurado). Chamado uma vez
        por treinamento, ao final da calibração."""

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
    ) -> None:
        """UPSERT do checkpoint: 1 linha por training_id (substitui quando Acc melhora)."""

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
    ) -> None:
        """Persiste accuracy, loss, f1_macro, τ_eff, per_class_f1 e round_duration_s por rodada.
        tau_effs é None por elemento quando o algoritmo é FedAvg."""

    @abstractmethod
    def load_latest(self) -> Optional[Dict]:
        """Retorna {'model_state': OrderedDict, 'vocab': dict} do checkpoint mais recente, ou None."""

    @abstractmethod
    def load_best(self, training_id: Optional[int] = None) -> Optional[Dict]:
        """Retorna o checkpoint com maior acurácia do treinamento indicado.
        Se training_id=None, usa o comportamento legado (melhor global — evitar)."""
