"""
schedule_state.py
Estado persistente do scheduler entre reinicializações.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Caminho padrão do arquivo de estado (ajuste conforme sua estrutura)
DEFAULT_STATE_PATH = Path("scheduler_state.json")


@dataclass
class SchedulerState:
    """Estado persistente do scheduler entre reinicializações."""
    last_run: Optional[str] = None           # ISO timestamp
    current_round: int = 0
    total_rounds_completed: int = 0
    client_history: Dict[str, List[str]] = field(default_factory=dict)
    accuracy_history: List[float] = field(default_factory=list)
    converged: bool = False
    convergence_round: Optional[int] = None

    def save(self, path: Path = DEFAULT_STATE_PATH):
        """Serializa o estado para JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "SchedulerState":
        """Carrega estado de JSON ou retorna estado inicial."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        return cls()