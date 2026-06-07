"""
inference_engine.py
Wrapper do modelo MOSAIC-FL para inferência single-patient em tempo real.

Separado do loop de treinamento federado (FedProxClient) — recebe a sequência
de exames de UM paciente e retorna P(risco_alto) em O(ms).

Tokenização: mapeamento determinístico por MD5 do nome do exame para evitar
dependência de um arquivo de vocabulário fixo.  Em produção, substituir por
um vocabulário alinhado ao treinamento.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Defaults para o caso de mosaicfl não estar instalado
_MOSAICFL_AVAILABLE = False
_VOCAB_SIZE = 10000
_MAX_SEQ_LEN = 128

try:
    from mosaicfl.v2.config import MODEL_CFG
    from mosaicfl.v2.model_v2 import SimplifiedBEHRT
    _MOSAICFL_AVAILABLE = True
    _VOCAB_SIZE = MODEL_CFG.vocab_size
    _MAX_SEQ_LEN = MODEL_CFG.max_seq_len
except Exception:
    pass

# Exportação pública — usada em testes e por consumidores externos
VOCAB_SIZE = _VOCAB_SIZE
MAX_SEQ_LEN = _MAX_SEQ_LEN


def exam_name_to_token(name: str) -> int:
    """MD5 do nome do exame → índice em [1, _VOCAB_SIZE-2].

    0 é PAD, _VOCAB_SIZE-1 é CLS — ambos reservados.
    """
    digest = int(hashlib.md5(name.upper().encode()).hexdigest(), 16)
    return (digest % (_VOCAB_SIZE - 2)) + 1


def records_to_tokens(records: list, seq_len: int = _MAX_SEQ_LEN) -> list[int]:
    """Converte lista de ExamRecord em sequência de tokens padded.

    Ordena por data para preservar ordem temporal na atenção do BEHRT.
    """
    sorted_records = sorted(records, key=lambda r: r.date)
    tokens = [exam_name_to_token(r.exam_name) for r in sorted_records]
    tokens = tokens[:seq_len]
    tokens += [0] * (seq_len - len(tokens))
    return tokens


class InferenceEngine:
    """Carrega o modelo treinado e expõe `predict(exam_records) → float`."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        if not _MOSAICFL_AVAILABLE:
            raise RuntimeError("mosaicfl não está instalado — execute 'pip install -e .'")

        self.model = SimplifiedBEHRT()
        self.model.eval()
        self._checkpoint_path: Optional[Path] = None

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load(Path(checkpoint_path))
        else:
            logger.warning(
                "inference_engine_no_checkpoint",
                extra={"path": str(checkpoint_path)},
            )

    def _load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self._checkpoint_path = path
        logger.info("inference_engine_loaded", extra={"path": str(path)})

    def reload(self, checkpoint_path: Path) -> None:
        """Recarrega pesos após novo round de treinamento federado."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self._load(checkpoint_path)

    def predict(self, exam_records: list) -> float:
        """Retorna P(risco_alto) ∈ [0.0, 1.0] para a sequência de exames.

        Args:
            exam_records: lista de ExamRecord (de integration/clinical-path/models.py)
        """
        if not exam_records:
            return 0.0

        tokens = records_to_tokens(exam_records)
        x = torch.tensor([tokens], dtype=torch.long)
        mask = x == 0  # True onde há padding

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, mask=mask)

        probs = F.softmax(logits, dim=-1)
        return float(probs[0, 1])  # P(classe positiva / alto risco)

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path
