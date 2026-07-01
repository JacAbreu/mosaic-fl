"""
term_manager — Gerencia o ciclo de vida dos termos em knowledge.term_dictionary.

Ver pending_workflow.py para o fluxo operacional completo (validação → revisão →
ativação/correção).

Submódulos:
  models.py              — PendingTerm, ValidationResult
  resolution.py            — _to_canonical, _load_alias_cache, _resolve_one
  pending_workflow.py         — validate_analytes_before_load, list_pending_terms,
                                activate_term, correct_term, activate_all_auto_normalized
"""
from .models import PendingTerm, ValidationResult
from .pending_workflow import (
    activate_all_auto_normalized,
    activate_term,
    correct_term,
    list_pending_terms,
    validate_analytes_before_load,
)

__all__ = [
    "PendingTerm",
    "ValidationResult",
    "validate_analytes_before_load",
    "list_pending_terms",
    "activate_term",
    "correct_term",
    "activate_all_auto_normalized",
]
