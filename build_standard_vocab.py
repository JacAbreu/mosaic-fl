"""
build_standard_vocab.py
Constrói o vocabulário padrão a partir das tabelas de conhecimento e o persiste em
checkpoints/standard_vocab.json (ou caminho configurável).

O vocabulário é determinístico — derivado exclusivamente de:
  - knowledge.term_dictionary  (analitos canônicos ativos)
  - knowledge.analyte_references  (existência de referências canônicas)

Não depende de dados de pacientes. Execute:
  1. Após migrate + populate_term_dictionary + compute_analyte_references
  2. Antes de iniciar qualquer rodada de treinamento federado
  3. Distribua o arquivo gerado a todos os clientes FL

Uso:
    python build_standard_vocab.py
    python build_standard_vocab.py --output models/vocab.json
    python build_standard_vocab.py --dry-run     # exibe stats, não salva
    python build_standard_vocab.py --token-mode FULL  (padrão)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Tokens especiais — devem ser idênticos a SequencePipeline._SPECIAL
_SPECIAL: dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}

_CLASSIFICATIONS_WITH_REFS = ("HIGH", "NORMAL", "LOW")


def _make_token(analyte: str, classification: str, mode: str = "FULL") -> str:
    """Espelha exatamente mosaicfl.core.preprocessor._make_token."""
    if mode == "ANALYTE_ONLY":
        return analyte
    if mode == "CLASS_ONLY":
        return classification
    if classification == "NO_REF":
        return analyte
    return f"{analyte}_{classification}"


def build_standard_vocab(db_url: str, token_mode: str = "FULL") -> dict[str, int]:
    """Constrói o vocabulário padrão a partir das tabelas de conhecimento.

    Cada analito ativo gera:
      - Com referências canônicas:  3 tokens (HIGH, NORMAL, LOW)
      - Sem referências:            1 token  (só o nome canônico — NO_REF)

    Returns:
        {token_str: token_id} — determinístico, ordenado alfabeticamente por analito.
    """
    engine = create_engine(db_url)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                td.canonical,
                ar.ref_low,
                ar.ref_high
            FROM knowledge.term_dictionary td
            LEFT JOIN knowledge.analyte_references ar
                   ON ar.canonical = td.canonical
                  AND ar.sex IS NULL
            WHERE td.term_type = 'analyte'
              AND td.active    = TRUE
            ORDER BY td.canonical
        """)).fetchall()

    vocab: dict[str, int] = dict(_SPECIAL)
    idx = len(_SPECIAL)

    analytes_with_refs = 0
    analytes_no_refs   = 0

    for row in rows:
        canonical = row.canonical
        has_refs = (
            row.ref_low is not None
            and not (float(row.ref_low) == 0.0 and float(row.ref_high) == 0.0)
        )

        if has_refs:
            for cls in _CLASSIFICATIONS_WITH_REFS:
                token = _make_token(canonical, cls, token_mode)
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
            analytes_with_refs += 1
        else:
            token = _make_token(canonical, "NO_REF", token_mode)
            if token not in vocab:
                vocab[token] = idx
                idx += 1
            analytes_no_refs += 1

    logger.info(
        "vocab construído: %d analitos com refs, %d sem refs, %d tokens totais",
        analytes_with_refs, analytes_no_refs, len(vocab),
    )
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói o vocabulário padrão do MOSAIC-FL")
    parser.add_argument(
        "--db-url",
        default=os.getenv("FL_DB_URL"),
        help="PostgreSQL connection string (padrão: $FL_DB_URL)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/standard_vocab.json",
        help="Caminho de saída do vocab JSON",
    )
    parser.add_argument(
        "--token-mode",
        default="FULL",
        choices=["FULL", "ANALYTE_ONLY", "CLASS_ONLY"],
        help="Modo de tokenização (deve ser igual ao usado no treinamento)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe estatísticas sem salvar o arquivo",
    )
    args = parser.parse_args()

    if not args.db_url:
        logger.error("--db-url ou $FL_DB_URL obrigatório")
        sys.exit(1)

    vocab = build_standard_vocab(args.db_url, token_mode=args.token_mode)

    logger.info("tokens especiais : %d", len(_SPECIAL))
    logger.info("tokens clínicos  : %d", len(vocab) - len(_SPECIAL))
    logger.info("vocab total      : %d tokens", len(vocab))

    if args.dry_run:
        logger.info("--dry-run: arquivo não salvo")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info("vocab salvo em: %s", out_path)


if __name__ == "__main__":
    main()
