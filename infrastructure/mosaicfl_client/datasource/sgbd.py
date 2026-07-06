"""sgbd.py — Fonte de dados SGBD (produção): PostgreSQL/MySQL/SQLite do hospital via SequencePipeline."""
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader, TensorDataset

from .base import DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, DEFAULT_VOCAB_SIZE, DataSource

logger = logging.getLogger(__name__)


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
                "Execute scripts/build_standard_vocab.py antes do treinamento federado em produção."
            )

        logger.info("[SGBD] Construindo sequências via SequencePipeline hospital=%s", self.hospital_id)
        pipeline = SequencePipeline(
            connection_string=self.connection_string,
            hospital_id=self.hospital_id or None,
            max_seq_len=self.seq_len,
            max_vocab_size=len(standard_vocab) if standard_vocab else self.vocab_size,
        )
        # build() retorna 5 valores (sequences, labels, vocab, demographics, dia_relativos).
        # FedProxClient.fit()/evaluate() (src/mosaicfl/core/client.py) iteram o loader como
        # (batch_x, batch_y, batch_dia) — 3 elementos, dia_relativo obrigatório (usado em
        # model(batch_x, dia_relativo=batch_dia)) — por isso precisa estar no TensorDataset.
        # demographics fica de fora: é usado só pela ablation study (late fusion), que tem seu
        # próprio client/loop de treino separado — não o FedProxClient usado neste datasource.
        sequences, labels, vocab, _demographics, dia_relativos = pipeline.build(vocab=standard_vocab)
        self.vocab = vocab
        self._n_sequences = len(sequences)

        dataset = TensorDataset(sequences, labels, dia_relativos)
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
