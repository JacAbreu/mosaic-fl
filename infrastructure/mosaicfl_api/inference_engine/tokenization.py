"""
tokenization.py — Resolução de termos e classificação (espelha a ingestão) + tokenização.

Tokenização alinhada ao treinamento:
  1. Resolve nome canônico do analito via knowledge.term_dictionary
  2. Classifica o valor via knowledge.analyte_references (HIGH/NORMAL/LOW/NO_REF)
  3. Compõe o token com o mesmo TokenMode usado no treinamento
  4. Mapeia para ID via vocabulário gravado junto com o checkpoint
"""
from .compat import MAX_SEQ_LEN as _MAX_SEQ_LEN
from .compat import _make_token


def _load_alias_cache(conn) -> dict[str, str]:
    """Carrega {normalize(alias): canonical} de knowledge.term_dictionary."""
    from sqlalchemy import text
    from integration.column_resolver import normalize

    rows = conn.execute(text("""
        SELECT canonical, alias
        FROM knowledge.term_dictionary
        WHERE term_type = 'analyte' AND active = TRUE
    """)).fetchall()

    cache: dict[str, str] = {}
    for canonical, alias in rows:
        cache[normalize(alias)]     = canonical
        cache[normalize(canonical)] = canonical
    return cache


def _load_canonical_refs(conn) -> dict[str, tuple[float, float]]:
    """Carrega {canonical: (ref_low, ref_high)} de knowledge.analyte_references."""
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT canonical, ref_low, ref_high
        FROM knowledge.analyte_references
        WHERE sex IS NULL
    """)).fetchall()
    return {r.canonical: (float(r.ref_low), float(r.ref_high)) for r in rows}


def _resolve_canonical(raw_name: str, alias_cache: dict[str, str]) -> str:
    """Resolve o nome canônico a partir do alias_cache. Fallback: normalize().upper().

    Usa apenas exact match — aliases verdadeiros devem estar em term_dictionary.
    Starts-with é omitido: prefixo compartilhado não implica equivalência clínica.
    """
    from integration.column_resolver import normalize

    norm = normalize(raw_name)
    if norm in alias_cache:
        return alias_cache[norm]
    return norm.upper()


def _classify(value: float, canonical: str, refs: dict[str, tuple[float, float]]) -> str:
    """Classifica o valor em relação às referências canônicas."""
    if canonical not in refs:
        return "NO_REF"
    ref_low, ref_high = refs[canonical]
    if ref_low == 0.0 and ref_high == 0.0:
        return "NO_REF"
    if value < ref_low:
        return "LOW"
    if value > ref_high:
        return "HIGH"
    return "NORMAL"


def records_to_tokens(
    records: list,
    vocab: dict[str, int],
    alias_cache: dict[str, str],
    canonical_refs: dict[str, tuple[float, float]],
    seq_len: int = _MAX_SEQ_LEN,
    token_mode: str = "FULL",
) -> list[int]:
    """Converte ExamRecord em sequência de IDs usando o vocabulário do treino.

    Segue exatamente o mesmo pipeline de tokenização do SequencePipeline:
      1. Resolve nome canônico
      2. Classifica o valor
      3. Compõe token com token_mode
      4. Mapeia para ID via vocab (UNK=1 se token fora do vocabulário)
    """
    unk_id = 1  # <UNK>
    sorted_records = sorted(records, key=lambda r: r.date)

    tokens: list[int] = []
    for r in sorted_records:
        canonical       = _resolve_canonical(r.exam_name, alias_cache)
        classification  = _classify(r.value, canonical, canonical_refs)
        token_str       = _make_token(canonical, classification, token_mode)
        token_id        = vocab.get(token_str, unk_id)
        tokens.append(token_id)

    tokens = tokens[:seq_len]
    tokens += [0] * (seq_len - len(tokens))  # padding com PAD=0
    return tokens
