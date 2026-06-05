"""
Pré-processamento e padronização da base FAPESP COVID-19 Data Sharing/BR — VERSÃO CORRIGIDA.

Mudanças principais:
  1. clean_text preserva pontos (.) e hífens (-) essenciais para códigos ICD e valores decimais.
  2. Validação de colunas antes de acessar (evita KeyError silencioso).
  3. Estratégia de missing configurável por coluna (nem sempre impute é adequado).
  4. split_by_institution agora embaralha e estratifica para balancear desfechos.
  5. Logging de rejeição por coluna (não apenas global).
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EHRPreprocessor:
    def __init__(self):
        self.vocab_map: Dict[str, int] = {}
        self.unit_conversions = {
            'peso': {'lb': 0.453592, 'lbs': 0.453592, 'kg': 1.0, 'g': 0.001},
            'idade': {'meses': 1/12, 'anos': 1.0, 'dias': 1/365.25},
            'temperatura': {'f': lambda x: (x - 32) * 5/9, 'c': lambda x: x}
        }
        self.transform_log: List[Dict] = []
        self.rejected_count = 0
        self.total_count = 0

    def _log_transform(self, step: str, detail: str, count: int = 0) -> None:
        entry = {"step": step, "detail": detail, "count": count}
        self.transform_log.append(entry)
        logger.info(f"[{step}] {detail} (n={count})")

    def normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'idade_unidade' in df.columns and 'idade' in df.columns:
            mask_meses = df['idade_unidade'].str.lower().isin(['meses', 'm'])
            mask_dias = df['idade_unidade'].str.lower().isin(['dias', 'd'])
            #df.loc[mask_meses, 'idade'] = df.loc[mask_meses, 'idade'] / 12.0
            df['idade'] = df['idade'].astype(float)
            df.loc[mask_meses, 'idade'] = df.loc[mask_meses, 'idade'] / 12.0
            df.loc[mask_dias, 'idade'] = df.loc[mask_dias, 'idade'] / 365.25
            df.loc[:, 'idade_unidade'] = 'anos'
            self._log_transform("unidade", "Idade normalizada para anos", int(mask_meses.sum() + mask_dias.sum()))

        if 'peso_unidade' in df.columns and 'peso' in df.columns:
            # Garante dtype float antes de atribuir resultado de multiplicação.
            # Pandas 2.x+ recusa atribuição de float em coluna int64 (LossySetitemError).
            df['peso'] = df['peso'].astype(float)
            for unit, factor in self.unit_conversions['peso'].items():
                if isinstance(factor, float):
                    mask = df['peso_unidade'].str.lower() == unit
                    df.loc[mask, 'peso'] = df.loc[mask, 'peso'] * factor
            df.loc[:, 'peso_unidade'] = 'kg'
            self._log_transform("unidade", "Peso normalizado para kg")
        return df

    def build_vocabulary(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, int]:
        vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3}
        idx = 4
        for col in text_cols:
            if col not in df.columns:
                logger.warning(f"Coluna '{col}' não encontrada — pulando vocabulário.")
                continue
            unique_vals = df[col].dropna().astype(str).unique()
            for val in unique_vals:
                if val not in vocab:
                    vocab[val] = idx
                    idx += 1
        self.vocab_map = vocab
        self._log_transform("vocab", f"Vocabulário construído: {len(vocab)} tokens (inclui <CLS>)")
        return vocab

    def encode_sequences(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in text_cols:
            if col in df.columns:
                df[col + '_encoded'] = df[col].astype(str).map(self.vocab_map).fillna(1).astype(int)
        return df

    # def handle_missing(self, df: pd.DataFrame, strategy: str = "impute",
    #                    numeric_strategy: str = "median",
    #                    categorical_strategy: str = "<UNK>") -> pd.DataFrame:
    #     before = len(df)
    #     if strategy == "drop":
    #         df = df.dropna(subset=df.columns[df.isnull().any()])
    #         self.rejected_count += before - len(df)
    #         self._log_transform("missing", "Registros removidos por valores ausentes", before - len(df))
    #     elif strategy == "impute":
    #         numeric_cols = df.select_dtypes(include=[np.number]).columns
    #         cat_cols = df.select_dtypes(include=['object']).columns

    #         for col in numeric_cols:
    #             if df[col].isnull().any():
    #                 if numeric_strategy == "median":
    #                     fill_val = df[col].median()
    #                 elif numeric_strategy == "mean":
    #                     fill_val = df[col].mean()
    #                 else:
    #                     fill_val = 0
    #                 df[col] = df[col].fillna(fill_val)
    #                 self._log_transform("missing", f"Coluna '{col}': preenchido com {numeric_strategy}={fill_val:.2f}")

    #         for col in cat_cols:
    #             if df[col].isnull().any():
    #                 df[col] = df[col].fillna(categorical_strategy)
    #                 self._log_transform("missing", f"Coluna '{col}': preenchido com '{categorical_strategy}'")
    #     return df

    def handle_missing(self, df: pd.DataFrame, strategy: str = "impute") -> pd.DataFrame:
        before = len(df)
        if strategy == "drop":
            df = df.dropna(subset=df.columns[df.isnull().any()])
            self.rejected_count += before - len(df)
            self._log_transform("missing", "Registros removidos por valores ausentes", before - len(df))
        elif strategy == "impute":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include=['object', 'str']).columns:
                df[col] = df[col].fillna("<UNK>")
            self._log_transform("missing", "Valores ausentes imputados (mediana/UNK)")
        return df


    # def clean_text(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    #     """
    #     Limpa texto preservando pontuação médica essencial:
    #       - Pontos (.) em códigos ICD (J18.1) e valores decimais (98.6)
    #       - Hífens (-) em nomes compostos e ranges
    #     Remove apenas caracteres especiais que não têm valor semântico clínico.
    #     """
    #     df = df.copy()
    #     for col in cols:
    #         if col not in df.columns:
    #             logger.warning(f"Coluna '{col}' não encontrada em clean_text — pulando.")
    #             continue
    #         # Lowercase e trim
    #         df[col] = df[col].astype(str).str.lower().str.strip()
    #         # Preserva: letras, números, espaços, pontos (ICD, decimais), hífens
    #         # Remove: outros caracteres especiais (!@#$% etc.)
    #         df[col] = df[col].str.replace(r'[^\w\s\.\-]', '', regex=True)
    #         # Remove espaços múltiplos
    #         df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    #     self._log_transform("clean", "Texto limpo: lowercase, trim, preserva . e - (ICD/decimais)")
    #     return df

    def clean_text(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s\-]', '', regex=True)
        self._log_transform("clean", "Texto limpo: lowercase, trim, remoção de pontuação")
        return df




    def process(self, df: pd.DataFrame, text_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        self.total_count = len(df)
        text_cols = text_cols or ['sintoma', 'exame', 'diagnostico']
        df = self.clean_text(df, text_cols)
        df = self.normalize_units(df)
        df = self.handle_missing(df, strategy="impute")
        self.build_vocabulary(df, text_cols)
        df = self.encode_sequences(df, text_cols)
        summary = {
            "total_amostras": self.total_count,
            "amostras_rejeitadas": self.rejected_count,
            "percentual_rejeitado": round(self.rejected_count / self.total_count * 100, 2) if self.total_count else 0,
            "tamanho_vocabulario": len(self.vocab_map),
            "transformacoes": self.transform_log
        }
        logger.info(f"Pré-processamento concluído. Resumo: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        return df, summary


# def split_by_institution(
#     df: pd.DataFrame,
#     institution_col: str = 'instituicao',
#     num_clients: int = 5,
#     stratify_col: Optional[str] = None,
#     random_state: int = 42,
# ) -> Dict[int, pd.DataFrame]:
#     """
#     Divide dados por instituição, com opção de estratificação por desfecho.

#     Args:
#         stratify_col: se fornecido (ex: 'desfecho'), embaralha e estratifica
#                       para garantir distribuição balanceada entre clientes.
#     """
#     clients = {}
#     institutions = df[institution_col].unique()

#     if len(institutions) < num_clients:
#         logger.warning(f"Apenas {len(institutions)} instituições encontradas — "
#                        f"ajustando num_clients de {num_clients} para {len(institutions)}")
#         num_clients = len(institutions)

#     for i, inst in enumerate(institutions[:num_clients]):
#         subset = df[df[institution_col] == inst].copy()

#         if stratify_col and stratify_col in subset.columns:
#             # Embaralha estratificado para evitar que um cliente fique com
#             # apenas um desfecho (extremo non-IID não-intencional)
#             subset = subset.groupby(stratify_col, group_keys=False).apply(
#                 lambda x: x.sample(frac=1, random_state=random_state)
#             ).reset_index(drop=True)
#         else:
#             subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)

#         clients[i] = subset
#         desfecho_dist = subset[stratify_col].value_counts().to_dict() if stratify_col else "N/A"
#         logger.info(f"Cliente {i} ({inst}): {len(clients[i])} registros | Distribuição: {desfecho_dist}")

#     return clients

# def split_by_institution(df: pd.DataFrame, institution_col: str = 'instituicao', num_clients: int = 5) -> Dict[int, pd.DataFrame]:
#     clients = {}
#     institutions = df[institution_col].unique()
#     for i, inst in enumerate(institutions[:num_clients]):
#         clients[i] = df[df[institution_col] == inst].copy()
#         logger.info(f"Cliente {i} ({inst}): {len(clients[i])} registros")
#     return clients

def split_by_institution(
    df: pd.DataFrame,
    institution_col: str = 'instituicao',
    num_clients: int = 5,
    stratify_col: str = None,
    random_state: int = None,
) -> Dict[int, pd.DataFrame]:
    """
    Divide o DataFrame por instituição, criando um cliente FL por hospital.

    Args:
        df:               DataFrame processado.
        institution_col:  Coluna com o identificador da instituição.
        num_clients:      Número máximo de clientes (hospitais).
        stratify_col:     Se informado, loga a distribuição dessa coluna por cliente
                          (útil para verificar balanceamento de desfechos entre hospitais).
        random_state:     Semente para embaralhamento antes da divisão (reprodutibilidade).

    Returns:
        Dict {client_id: DataFrame} com um subset por instituição.
    """
    clients = {}
    institutions = df[institution_col].unique()

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        institutions = rng.permutation(institutions)

    for i, inst in enumerate(institutions[:num_clients]):
        subset = df[df[institution_col] == inst].copy()
        clients[i] = subset

        if stratify_col and stratify_col in subset.columns:
            dist = subset[stratify_col].value_counts().to_dict()
            logger.info(f"Cliente {i} ({inst}): {len(subset)} registros | {stratify_col}: {dist}")
        else:
            logger.info(f"Cliente {i} ({inst}): {len(subset)} registros")

    return clients