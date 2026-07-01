"""
postprocessing.py — Normalização pós-carregamento (independente da fonte de dados).

_map_columns                    — schema real → nomes padronizados internos
_validate_schema                 — valida colunas obrigatórias após mapeamento
_convert_desfecho                 — desfecho textual → numérico
_compute_idade_from_nascimento     — deriva idade a partir de data de nascimento
_generate_synthetic_fallback        — dados sintéticos para testes/desenvolvimento
"""
import logging
import random

import numpy as np
import pandas as pd

from ..config import FED_CFG
from .settings import COLUMN_MAPPING, DESFECHO_TEXT_TO_NUMERIC

logger = logging.getLogger(__name__)


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas do schema real → nomes padronizados internos."""
    rename_map = {}
    csv_columns_lower = {c.lower().strip(): c for c in df.columns}

    for standard_name, candidates in COLUMN_MAPPING.items():
        for candidate in candidates:
            candidate_clean = candidate.lower().strip()
            if candidate_clean in csv_columns_lower:
                original_name = csv_columns_lower[candidate_clean]
                rename_map[original_name] = standard_name
                logger.info(f"Coluna mapeada: '{original_name}' → '{standard_name}'")
                break
        else:
            logger.warning(f"Coluna não encontrada: '{standard_name}' (tentados: {candidates})")

    return df.rename(columns=rename_map)


def _validate_schema(df: pd.DataFrame) -> None:
    """Valida colunas obrigatórias após mapeamento."""
    required = ["instituicao", "desfecho"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        available = [c for c in df.columns if not c.startswith("Unnamed")]
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing}\n"
            f"Colunas disponíveis: {available}\n\n"
            f"DICA: Edite COLUMN_MAPPING em settings.py para mapear "
            f"os nomes reais do seu schema (CSV ou SGBD)."
        )

    if df["desfecho"].dtype == object:
        unique_vals = df["desfecho"].dropna().unique()[:20]
        logger.warning(
            f"'desfecho' é textual. Primeiros valores: {list(unique_vals)}. "
            f"Aplicando mapeamento DESFECHO_TEXT_TO_NUMERIC."
        )


def _convert_desfecho(df: pd.DataFrame) -> pd.DataFrame:
    """Converte desfecho textual → numérico."""
    # Em pandas 4.x, colunas de string têm dtype `str` (StringDtype), não `object`.
    # Usar is_numeric_dtype é robusto para qualquer versão do pandas.
    if pd.api.types.is_numeric_dtype(df["desfecho"]):
        return df

    df = df.copy()
    df["desfecho_original"] = df["desfecho"]
    df["desfecho"] = df["desfecho"].map(DESFECHO_TEXT_TO_NUMERIC)

    n_unmapped = df["desfecho"].isna().sum()
    if n_unmapped > 0:
        unmapped = df.loc[df["desfecho"].isna(), "desfecho_original"].unique()[:10]
        logger.warning(
            f"{n_unmapped} desfechos não mapeados. Valores: {list(unmapped)}. "
            f"Adicione em DESFECHO_TEXT_TO_NUMERIC."
        )

    return df


def _compute_idade_from_nascimento(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula idade a partir da data de nascimento, se disponível."""
    if "idade" in df.columns:
        return df

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["nasc", "birth", "dt_nasc"])]
    if not date_cols:
        return df

    birth_col = date_cols[0]
    logger.info(f"'idade' ausente — calculando a partir de '{birth_col}'")

    try:
        df = df.copy()
        birth_dates = pd.to_datetime(df[birth_col], errors="coerce", dayfirst=True)
        today = pd.Timestamp.now()
        df["idade"] = ((today - birth_dates).dt.days / 365.25).round(1)
        df["idade_unidade"] = "anos"
        logger.info(f"Idade calculada para {df['idade'].notna().sum()} registros.")
    except Exception as e:
        logger.warning(f"Erro ao calcular idade: {e}")

    return df


def _generate_synthetic_fallback(n_samples: int = 1000) -> pd.DataFrame:
    """Gera dados sintéticos para testes. NUNCA use em produção."""
    random.seed(FED_CFG.random_seed)
    np.random.seed(FED_CFG.random_seed)

    institutions = ["HC_usp", "HCFMUSP", "Hospital_Clinicas_SP", "Incor", "Hospital_Sirio"]
    sintomas = ["febre", "tosse", "dispneia", "fadiga", "mialgia", "cefaleia", "anosmia", "diarreia"]
    exames = ["rt_pcr_positivo", "tomografia_normal", "tomografia_vidro_fosco", "rx_consolidacao"]
    diagnosticos = ["covid19_leve", "covid19_moderado", "covid19_grave", "pneumonia_bacteriana"]

    df = pd.DataFrame({
        "instituicao": np.random.choice(institutions, n_samples),
        "idade": np.random.randint(18, 90, n_samples),
        "idade_unidade": np.random.choice(["anos", "meses"], n_samples, p=[0.95, 0.05]),
        "peso": np.random.uniform(50, 120, n_samples),
        "peso_unidade": np.random.choice(["kg", "lb"], n_samples, p=[0.9, 0.1]),
        "temperatura": np.random.uniform(36.0, 40.0, n_samples),
        "sintoma": np.random.choice(sintomas, n_samples),
        "exame": np.random.choice(exames, n_samples),
        "diagnostico": np.random.choice(diagnosticos, n_samples),
        "desfecho": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })

    logger.warning(f"[FALLBACK SINTÉTICO] {n_samples} registros gerados — NÃO SÃO REAIS!")
    return df
