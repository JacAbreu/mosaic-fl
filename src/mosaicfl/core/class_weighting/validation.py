"""
validation.py — Validação de overrides de peso de classe, compartilhada entre todo
ponto de escrita (endpoint web /api/admin/orchestration-config/class-weights e
scripts/set_class_weight_overrides.py) — mesma regra, um lugar só de verdade.
"""
from typing import Dict, Sequence


def validate_overrides(overrides: Dict[str, float], class_labels: Sequence[str]) -> None:
    """Levanta ValueError com mensagem descritiva se `overrides` for inválido.
    Não escreve nada — só valida. Classe desconhecida ou peso <= 0 são erros de
    configuração, não deveriam nunca chegar no banco."""
    unknown = sorted(set(overrides.keys()) - set(class_labels))
    if unknown:
        raise ValueError(f"Classe(s) desconhecida(s): {unknown}. Válidas: {list(class_labels)}")
    invalid = {name: w for name, w in overrides.items() if w <= 0}
    if invalid:
        raise ValueError(f"Peso precisa ser > 0: {invalid}")
