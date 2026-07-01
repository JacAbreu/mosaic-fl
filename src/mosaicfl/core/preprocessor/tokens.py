"""
tokens.py — Composição de tokens clínicos para o vocabulário do BEHRT.
"""


class TokenMode:
    """Modos de composição de token para o pipeline de treino.

    FULL            — analito + classificação: LEUCOCITOS_HIGH  (padrão)
    ANALYTE_ONLY    — apenas o analito:        LEUCOCITOS
    CLASS_ONLY      — apenas a classificação:  HIGH

    Permite experimentar diferentes hipóteses de treino sem recarregar dados:
    - FULL:         o nível clínico importa junto com o analito
    - ANALYTE_ONLY: basta saber que o exame foi solicitado (perfil de investigação)
    - CLASS_ONLY:   padrão de anormalidade independente do analito
    """
    FULL         = "FULL"
    ANALYTE_ONLY = "ANALYTE_ONLY"
    CLASS_ONLY   = "CLASS_ONLY"


def _make_token(analyte: str, classification: str, mode: str = TokenMode.FULL) -> str:
    """Gera token a partir do analito canônico e sua classificação clínica.

    analyte        — nome canônico em maiúsculas (ex: LEUCOCITOS)
    classification — HIGH | NORMAL | LOW | NO_REF (gravado em exam_records)
    mode           — TokenMode.FULL | ANALYTE_ONLY | CLASS_ONLY
    """
    if mode == TokenMode.ANALYTE_ONLY:
        return analyte
    if mode == TokenMode.CLASS_ONLY:
        return classification
    # FULL: sem referência disponível retorna só o analito
    if classification == "NO_REF":
        return analyte
    return f"{analyte}_{classification}"
