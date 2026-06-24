"""
datasource.py
Strategy Pattern para fontes de dados do cliente federado.

Suporta:
  - simulated: dados sintéticos para TCC/prototipagem
  - sgbd: conexão PostgreSQL/MySQL/SQLite do hospital
  - csv: arquivo CSV local (fallback para hospitais sem SGBD)
  - fhir: integração HL7 FHIR (futuro)

Uso:
    from datasource import DataSourceFactory

    source = DataSourceFactory.create("sgbd", connection_string="postgresql://...")
    loader = source.load()
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = int(os.getenv("FL_BATCH_SIZE", "16"))
DEFAULT_SEQ_LEN = int(os.getenv("FL_SEQ_LEN", "128"))
DEFAULT_VOCAB_SIZE = int(os.getenv("FL_VOCAB_SIZE", "10000"))
DEFAULT_NUM_SAMPLES = int(os.getenv("FL_SIM_SAMPLES", "200"))


# ════════════════════════════════════════════════════════════════════════════
# Interface base
# ════════════════════════════════════════════════════════════════════════════
class DataSource(ABC):
    """Interface para todas as fontes de dados do cliente."""

    @abstractmethod
    def load(self) -> DataLoader:
        """Retorna DataLoader PyTorch pronto para treinamento."""
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """Retorna metadados sobre a fonte (nome, tipo, registros, etc.)."""
        pass

    def validate(self) -> Tuple[bool, str]:
        """Valida se a fonte está acessível. Retorna (ok, mensagem)."""
        return True, "Validação padrão: OK"


# ════════════════════════════════════════════════════════════════════════════
# 1. Simulated — Dados sintéticos para TCC/prototipagem
# ════════════════════════════════════════════════════════════════════════════
class SimulatedDataSource(DataSource):
    """
    Gera dados sintéticos para desenvolvimento e testes.
    Útil quando não há acesso ao SGBD real ou para benchmark.
    """

    def __init__(
        self,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_len: int = DEFAULT_SEQ_LEN,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        num_classes: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 42,
        hospital_id: Optional[str] = None,  # aceito mas ignorado — sem partição em modo sintético
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.seed = seed

    def load(self) -> DataLoader:
        logger.info(
            f"[SIMULATED] Gerando {self.num_samples} amostras sintéticas "
            f"(seq_len={self.seq_len}, vocab_size={self.vocab_size})"
        )
        torch.manual_seed(self.seed)

        X = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_len))
        y = torch.randint(0, self.num_classes, (self.num_samples,))

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger.info(f"[SIMULATED] DataLoader pronto: {len(loader)} batches")
        return loader

    def get_metadata(self) -> dict:
        return {
            "type": "simulated",
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
        }

    def validate(self) -> Tuple[bool, str]:
        return True, "Fonte simulada: sempre disponível"


# ---------------------------------------------------------------------------
# Vocab padrão — carregado uma vez por processo
# ---------------------------------------------------------------------------

def _load_standard_vocab() -> Optional[dict]:
    """Tenta carregar o vocabulário pré-compartilhado do servidor.

    Caminho resolvido na ordem:
      1. $FL_VOCAB_PATH (configuração explícita)
      2. checkpoints/standard_vocab.json (padrão local)

    Retorna None se o arquivo não existir — SGBDDataSource emitirá um aviso
    e construirá o vocab localmente (adequado apenas para simulação).
    """
    candidates = [
        os.getenv("FL_VOCAB_PATH"),
        "checkpoints/standard_vocab.json",
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("[SGBD] Erro ao carregar vocab de %s: %s", path, exc)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 2. SGBD — PostgreSQL, MySQL, SQLite, SQL Server, Oracle
# ════════════════════════════════════════════════════════════════════════════
class SGBDDataSource(DataSource):
    """
    Fonte de dados para o cliente FL em produção.

    Usa SequencePipeline para construir sequências temporais de exames
    diretamente do banco PostgreSQL local do hospital, produzindo tensores
    Long compatíveis com SimplifiedBEHRT.

    Variáveis de ambiente:
      FL_DB_URL     — connection string PostgreSQL do hospital
      FL_CLIENT_ID  — hospital_id (ex: HSL, BPSP) para filtrar registros locais
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        hospital_id: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ):
        self.connection_string = connection_string or os.getenv("FL_DB_URL", "")
        self.hospital_id = hospital_id or os.getenv("FL_CLIENT_ID", "")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.vocab: dict = {}
        self._n_sequences: int = 0

    def validate(self) -> Tuple[bool, str]:
        if not self.connection_string:
            return False, "Connection string não configurada. Defina FL_DB_URL."
        if not self.hospital_id:
            return False, "ID do hospital não configurado. Defina FL_CLIENT_ID."
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            return True, f"Conectado: hospital={self.hospital_id} dialect={engine.dialect.name}"
        except ImportError:
            return False, "SQLAlchemy não instalado. Execute: pip install sqlalchemy[postgresql]"
        except Exception as e:
            return False, f"Erro de conexão: {e}"

    def load(self) -> DataLoader:
        from mosaicfl.core.preprocessor import SequencePipeline

        standard_vocab = _load_standard_vocab()
        if standard_vocab:
            logger.info(
                "[SGBD] vocab padrão carregado: %d tokens — aggregação federada válida",
                len(standard_vocab),
            )
        else:
            logger.warning(
                "[SGBD] standard_vocab.json não encontrado — vocab será construído localmente. "
                "Execute build_standard_vocab.py antes do treinamento federado em produção."
            )

        logger.info("[SGBD] Construindo sequências via SequencePipeline hospital=%s", self.hospital_id)
        pipeline = SequencePipeline(
            connection_string=self.connection_string,
            hospital_id=self.hospital_id or None,
            max_seq_len=self.seq_len,
            max_vocab_size=len(standard_vocab) if standard_vocab else self.vocab_size,
        )
        sequences, labels, vocab = pipeline.build(vocab=standard_vocab)
        self.vocab = vocab
        self._n_sequences = len(sequences)

        dataset = TensorDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        logger.info(
            "[SGBD] DataLoader pronto: %d sequências, %d batches, vocab_size=%d",
            self._n_sequences, len(loader), len(self.vocab),
        )
        return loader

    def get_metadata(self) -> dict:
        return {
            "type":        "sgbd",
            "hospital_id": self.hospital_id,
            "connection":  self.connection_string.split("@")[-1] if "@" in self.connection_string else "N/A",
            "sequences":   self._n_sequences,
            "vocab_size":  len(self.vocab),
            "batch_size":  self.batch_size,
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. CSV — Arquivo local (fallback para hospitais sem SGBD)
# ════════════════════════════════════════════════════════════════════════════
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

    def load(self) -> DataLoader:
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


# ════════════════════════════════════════════════════════════════════════════
# 4. Factory — Cria a fonte correta via configuração
# ════════════════════════════════════════════════════════════════════════════
class DataSourceFactory:
    """Factory para criar a fonte de dados correta via variável de ambiente."""

    _registry = {
        "simulated": SimulatedDataSource,
        "sgbd": SGBDDataSource,
        "csv": CSVDataSource,
    }

    @classmethod
    def create(cls, source_type: Optional[str] = None, **kwargs) -> DataSource:
        """
        Cria uma fonte de dados.

        Args:
            source_type: 'simulated', 'sgbd', 'csv'. Se None, lê FL_DATA_SOURCE.
            **kwargs: Parâmetros específicos da fonte.

        Returns:
            Instância de DataSource pronta para uso.

        Raises:
            ValueError: Se o tipo não for suportado.
        """
        source_type = (source_type or os.getenv("FL_DATA_SOURCE", "simulated")).lower().strip()

        if source_type not in cls._registry:
            raise ValueError(
                f"Fonte de dados '{source_type}' não suportada. "
                f"Opções: {list(cls._registry.keys())}"
            )

        source_class = cls._registry[source_type]
        instance = source_class(**kwargs)

        # Validação automática
        ok, msg = instance.validate()
        if not ok:
            logger.error(f"[FACTORY] Validação falhou: {msg}")
            raise RuntimeError(f"Fonte de dados inválida: {msg}")

        logger.info(f"[FACTORY] Fonte criada: {source_type} — {msg}")
        return instance

    @classmethod
    def register(cls, name: str, source_class: type):
        """Registra uma nova fonte de dados (extensibilidade)."""
        cls._registry[name] = source_class
        logger.info(f"[FACTORY] Fonte registrada: {name}")

    @classmethod
    def available_sources(cls) -> list:
        """Lista fontes disponíveis."""
        return list(cls._registry.keys())