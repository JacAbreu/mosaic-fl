"""
state_store.py — Persistência do estado de treinamento entre sessões do ServerApp.

Permite que o ServerApp detecte interrupções, restaure o ConvergenceTracker e
recarregue os pesos do último checkpoint antes de iniciar uma nova rodada.

Ciclo de vida do status:
    "pending"     → nunca rodou
    "running"     → sessão em andamento; se encontrado no load = foi interrompida
    "completed"   → convergência atingida normalmente
    "interrupted" → gravado no shutdown/crash (melhor esforço)
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

TrainingStatus = Literal["pending", "running", "completed", "interrupted"]

_VALID_STATUSES = {"pending", "running", "completed", "interrupted"}


@dataclass
class TrainingState:
    status: TrainingStatus = "pending"
    # Identifica a que "flwr run" este estado pertence — sem isso, um novo run
    # (run_id diferente) restaura convergência/checkpoint de um run anterior
    # não relacionado, mesmo que ele tenha falhado (ex: accuracy=0.0 sempre,
    # falsamente detectado como "convergido"). None = estado de formato antigo,
    # sem run_id gravado (trata como não-restaurável).
    run_id: Optional[int] = None
    last_round: int = 0
    # Histórico completo de accuracy — suficiente para restaurar ConvergenceTracker
    convergence_history: List[float] = field(default_factory=list)
    converged_round: Optional[int] = None
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    last_checkpoint: Optional[str] = None
    # Rounds que ultrapassaram o timeout na sessão atual
    timed_out_rounds: List[int] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainingStateStore:
    """Salva e carrega TrainingState em JSON. Thread-safe para leitura/escrita."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> TrainingState:
        """
        Carrega estado salvo. Se o arquivo não existir, retorna estado inicial.
        Se o status salvo era 'running', significa que a sessão foi interrompida.
        """
        if not self._path.exists():
            logger.info("training_state_not_found", extra={"path": str(self._path)})
            return TrainingState()

        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)

            # Filtra chaves desconhecidas para compatibilidade futura
            known = TrainingState.__dataclass_fields__
            filtered = {k: v for k, v in data.items() if k in known}

            # Garante status válido
            if filtered.get("status") not in _VALID_STATUSES:
                filtered["status"] = "interrupted"

            state = TrainingState(**filtered)

            if state.status == "running":
                state.status = "interrupted"
                logger.warning(
                    "training_state_interrupted",
                    extra={
                        "last_round": state.last_round,
                        "last_checkpoint": state.last_checkpoint,
                    },
                )
            else:
                logger.info(
                    "training_state_recovered",
                    extra={
                        "status": state.status,
                        "last_round": state.last_round,
                        "converged_round": state.converged_round,
                    },
                )

            return state

        except Exception as exc:
            logger.error("training_state_load_error", extra={"error": str(exc)})
            return TrainingState()

    def save(self, state: TrainingState) -> None:
        """Persiste estado no disco. Erros de I/O são logados mas não propagados."""
        state.updated_at = datetime.now().isoformat()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2)
        except Exception as exc:
            logger.error("training_state_save_error", extra={"error": str(exc)})
