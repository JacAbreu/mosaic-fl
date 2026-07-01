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
    ) -> int:
        """Registra um novo treinamento antes do loop FL. Retorna training_id."""

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
    ) -> None:
        """Atualiza fl_trainings com resultado final ao término do loop FL."""

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
