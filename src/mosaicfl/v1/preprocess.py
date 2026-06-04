"""
Pré-processamento e padronização da base FAPESP COVID-19 Data Sharing/BR.
Mapeia desafios de interoperabilidade entre instituições (HF1-HF5).
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Tuple, Dict, List
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

    def _log_transform(self, step: str, detail: str, count: int = 0):
        entry = {"step": step, "detail": detail, "count": count}
        self.transform_log.append(entry)
        logger.info(f"[{step}] {detail} (n={count})")

    def normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'idade_unidade' in df.columns and 'idade' in df.columns:
            mask_meses = df['idade_unidade'].str.lower().isin(['meses', 'm'])
            mask_dias = df['idade_unidade'].str.lower().isin(['dias', 'd'])
            df.loc[mask_meses, 'idade'] = df.loc[mask_meses, 'idade'] / 12.0
            df.loc[mask_dias, 'idade'] = df.loc[mask_dias, 'idade'] / 365.25
            df.loc[:, 'idade_unidade'] = 'anos'
            self._log_transform("unidade", "Idade normalizada para anos", int(mask_meses.sum() + mask_dias.sum()))

        if 'peso_unidade' in df.columns and 'peso' in df.columns:
            for unit, factor in self.unit_conversions['peso'].items():
                if isinstance(factor, float):
                    mask = df['peso_unidade'].str.lower() == unit
                    df.loc[mask, 'peso'] = df.loc[mask, 'peso'] * factor
            df.loc[:, 'peso_unidade'] = 'kg'
            self._log_transform("unidade", "Peso normalizado para kg")
        return df

    def build_vocabulary(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, int]:
        vocab = {"<<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
        idx = 3
        for col in text_cols:
            if col in df.columns:
                unique_vals = df[col].dropna().astype(str).unique()
                for val in unique_vals:
                    if val not in vocab:
                        vocab[val] = idx
                        idx += 1
        self.vocab_map = vocab
        self._log_transform("vocab", f"Vocabulário construído: {len(vocab)} tokens")
        return vocab

    def encode_sequences(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in text_cols:
            if col in df.columns:
                df[col + '_encoded'] = df[col].astype(str).map(self.vocab_map).fillna(1).astype(int)
        return df

    def handle_missing(self, df: pd.DataFrame, strategy: str = "impute") -> pd.DataFrame:
        before = len(df)
        if strategy == "drop":
            df = df.dropna(subset=df.columns[df.isnull().any()])
            self.rejected_count += before - len(df)
            self._log_transform("missing", "Registros removidos por valores ausentes", before - len(df))
        elif strategy == "impute":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna("<UNK>")
            self._log_transform("missing", "Valores ausentes imputados (mediana/UNK)")
        return df

    def clean_text(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s\-]', '', regex=True)
        self._log_transform("clean", "Texto limpo: lowercase, trim, remoção de pontuação")
        return df

    def process(self, df: pd.DataFrame, text_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
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


def split_by_institution(df: pd.DataFrame, institution_col: str = 'instituicao', num_clients: int = 5) -> Dict[int, pd.DataFrame]:
    clients = {}
    institutions = df[institution_col].unique()
    for i, inst in enumerate(institutions[:num_clients]):
        clients[i] = df[df[institution_col] == inst].copy()
        logger.info(f"Cliente {i} ({inst}): {len(clients[i])} registros")
    return clients
