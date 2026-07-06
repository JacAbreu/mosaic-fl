"""csv_source.py — Fonte de dados CSV local (fallback para hospitais sem SGBD)."""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import DEFAULT_BATCH_SIZE, DataSource

logger = logging.getLogger(__name__)


class CSVDataSource(DataSource):
    """
    Lê arquivo CSV local.
    Útil para hospitais que exportam dados periodicamente.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        sep: str = ",",
        encoding: str = "utf-8",
        batch_size: int = DEFAULT_BATCH_SIZE,
        hospital_id: Optional[str] = None,  # aceito mas ignorado — CSV já contém dados de um hospital
    ):
        self.filepath = filepath or os.getenv("FL_CSV_PATH", "data/hospital.csv")
        self.sep = sep
        self.encoding = encoding
        self.batch_size = batch_size
        self._df: Optional[pd.DataFrame] = None

    def validate(self) -> Tuple[bool, str]:
        path = Path(self.filepath)
        if not path.exists():
            return False, f"Arquivo não encontrado: {self.filepath}"
        if path.stat().st_size == 0:
            return False, f"Arquivo vazio: {self.filepath}"
        return True, f"Arquivo CSV: {path.stat().st_size / 1024:.1f} KB"

    def load(self, vocab: Optional[dict] = None) -> DataLoader:
        logger.info(f"[CSV] Lendo {self.filepath}...")
        df = pd.read_csv(self.filepath, sep=self.sep, encoding=self.encoding)
        self._df = df

        # Placeholder: mesma lógica de preprocess do SGBD
        # Em produção: usar EHRPreprocessor do pacote core
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        target_col = None
        for col in ["desfecho", "target", "label", "outcome"]:
            if col in df.columns:
                target_col = col
                break
        if target_col is None and len(numeric_cols) > 1:
            target_col = numeric_cols[-1]

        feature_cols = [c for c in numeric_cols if c != target_col]
        X = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float32)
        y = torch.tensor(df[target_col].fillna(0).values, dtype=torch.long) if target_col else torch.zeros(len(df), dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"[CSV] DataLoader pronto: {len(loader)} batches")
        return loader

    def get_metadata(self) -> dict:
        return {
            "type": "csv",
            "filepath": self.filepath,
            "records": len(self._df) if self._df is not None else 0,
            "batch_size": self.batch_size,
        }
