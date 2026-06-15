"""
inference_engine.py
Wrapper do modelo MOSAIC-FL para inferência single-patient em tempo real.

Tokenização alinhada ao treinamento:
  1. Resolve nome canônico do analito via knowledge.term_dictionary
  2. Classifica o valor via knowledge.analyte_references (HIGH/NORMAL/LOW/NO_REF)
  3. Compõe o token com o mesmo TokenMode usado no treinamento
  4. Mapeia para ID via vocabulário gravado junto com o checkpoint

Substituição do MD5: a tokenização por hash gerava tokens fora do vocabulário
treinado, tornando o predict() clinicamente inválido.
"""
import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MOSAICFL_AVAILABLE = False
_VOCAB_SIZE = 10000
_MAX_SEQ_LEN = 128

try:
    from mosaicfl.core.config import MODEL_CFG
    from mosaicfl.core.model import SimplifiedBEHRT
    from mosaicfl.core.preprocessor import TokenMode, _make_token
    _MOSAICFL_AVAILABLE = True
    _VOCAB_SIZE = MODEL_CFG.vocab_size
    _MAX_SEQ_LEN = MODEL_CFG.max_seq_len
except Exception:
    # Fallback local — espelha exatamente a lógica do preprocessor.py
    def _make_token(analyte: str, classification: str, mode: str = "FULL") -> str:
        if mode == "ANALYTE_ONLY":
            return analyte
        if mode == "CLASS_ONLY":
            return classification
        if classification == "NO_REF":
            return analyte
        return f"{analyte}_{classification}"

VOCAB_SIZE  = _VOCAB_SIZE
MAX_SEQ_LEN = _MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# Resolução de termos e classificação (espelha a ingestão)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Carrega o modelo treinado e expõe `predict(exam_records) → float`.

    O vocabulário é carregado do checkpoint — deve ser o mesmo gerado pelo
    SequencePipeline no treinamento. A tokenização usa os mesmos passos
    (resolução canônica → classificação → token_mode) para garantir
    consistência entre treino e inferência.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        db_url: Optional[str] = None,
        token_mode: str = "FULL",
    ):
        if not _MOSAICFL_AVAILABLE:
            raise RuntimeError("mosaicfl não está instalado — execute 'pip install -e .'")

        self.model = SimplifiedBEHRT()
        self.model.eval()
        self._checkpoint_path: Optional[Path] = None
        self.token_mode = token_mode

        self._vocab:         dict[str, int]           = {}
        self._alias_cache:   dict[str, str]           = {}
        self._canonical_refs: dict[str, tuple[float, float]] = {}
        self._mc_lock = threading.Lock()

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load(Path(checkpoint_path))

        if db_url:
            self._load_references(db_url)
        else:
            logger.warning(
                "inference_engine_no_db — tokenização sem resolução canônica; "
                "tokens podem não corresponder ao vocabulário treinado"
            )

    def _load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Checkpoint deve conter 'model_state' e 'vocab'
        if isinstance(state, dict) and "vocab" in state:
            self._vocab = state["vocab"]
            self.model.load_state_dict(state["model_state"])
        else:
            # Compatibilidade com checkpoints antigos (só pesos)
            self.model.load_state_dict(state)
            logger.warning(
                "inference_engine_legacy_checkpoint — vocabulário ausente; "
                "salve o vocab junto com os pesos do modelo"
            )
        self._checkpoint_path = path
        logger.info("inference_engine_loaded path=%s vocab_size=%d", path, len(self._vocab))

    def _load_references(self, db_url: str) -> None:
        """Carrega alias cache e refs canônicas do banco."""
        from sqlalchemy import create_engine

        engine = create_engine(db_url)
        with engine.connect() as conn:
            self._alias_cache    = _load_alias_cache(conn)
            self._canonical_refs = _load_canonical_refs(conn)

        logger.info(
            "inference_engine_references_loaded aliases=%d refs=%d",
            len(self._alias_cache), len(self._canonical_refs),
        )

    def reload(self, checkpoint_path: Path) -> None:
        """Recarrega pesos e vocabulário após novo round federado."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self._load(checkpoint_path)

    def reload_references(self, db_url: str) -> None:
        """Recarrega alias cache e refs canônicas — use quando analyte_references mudar."""
        self._load_references(db_url)

    def _tokenize(self, exam_records: list) -> "tuple[torch.Tensor, torch.Tensor]":
        tokens = records_to_tokens(
            records        = exam_records,
            vocab          = self._vocab,
            alias_cache    = self._alias_cache,
            canonical_refs = self._canonical_refs,
            seq_len        = _MAX_SEQ_LEN,
            token_mode     = self.token_mode,
        )
        x = torch.tensor([tokens], dtype=torch.long)
        return x, (x == 0)

    def predict_proba(self, exam_records: list, mc_samples: int = 50) -> dict:
        """Retorna probabilidades por classe com incerteza via MC Dropout.

        Returns:
            probabilities:   {label: {"value": float, "uncertainty": float}}
            predicted_class: índice da classe com maior probabilidade média
            predicted_label: label correspondente
            mc_samples:      número de amostras usadas
            risk_score:      escalar ponderado [0,1] para armazenamento histórico
        """
        n      = MODEL_CFG.num_classes if _MOSAICFL_AVAILABLE else 5
        labels = list(MODEL_CFG.class_labels) if _MOSAICFL_AVAILABLE else [f"class_{i}" for i in range(n)]
        empty  = {"value": 0.0, "uncertainty": 0.0}

        if not exam_records or not self._vocab:
            if not exam_records:
                logger.debug("predict_proba chamado com lista vazia")
            else:
                logger.warning("inference_engine_empty_vocab")
            return {
                "probabilities":   {l: dict(empty) for l in labels},
                "predicted_class": 0,
                "predicted_label": labels[0],
                "mc_samples":      0,
                "risk_score":      0.0,
                "trained":         self._checkpoint_path is not None,
            }

        x, mask = self._tokenize(exam_records)

        # Lock garante que model.train()/eval() não colidem entre requests simultâneos
        with self._mc_lock:
            self.model.train()
            all_probs: list = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    logits = self.model(x, mask=mask)
                    all_probs.append(F.softmax(logits, dim=-1)[0])
            self.model.eval()

        stacked    = torch.stack(all_probs, dim=0)   # (mc_samples, num_classes)
        mean_probs = stacked.mean(dim=0)              # (num_classes,)
        std_probs  = stacked.std(dim=0)               # (num_classes,)

        predicted_class = int(mean_probs.argmax())
        weights = torch.linspace(0.0, 1.0, n, device=mean_probs.device)
        risk_score = float((mean_probs * weights).sum())

        return {
            "probabilities": {
                label: {
                    "value":       round(float(mean_probs[i]), 4),
                    "uncertainty": round(float(std_probs[i]),  4),
                }
                for i, label in enumerate(labels)
            },
            "predicted_class": predicted_class,
            "predicted_label": labels[predicted_class],
            "mc_samples":      mc_samples,
            "risk_score":      round(risk_score, 4),
            "trained":         self._checkpoint_path is not None,
        }

    def predict(self, exam_records: list) -> float:
        """Escalar de risco ∈ [0,1] — mantido para compatibilidade com armazenamento histórico."""
        return self.predict_proba(exam_records)["risk_score"]

    def resolve_for_ingest(
        self,
        exam_name: str,
        value: float,
    ) -> "tuple[str, Optional[str], Optional[float], Optional[float]]":
        """Resolve nome canônico e classifica um exame para persistência.

        Returns:
            (canonical, classification, canonical_ref_low, canonical_ref_high)
            classification é None quando não há referência canônica disponível.
        """
        canonical = _resolve_canonical(exam_name, self._alias_cache)
        if canonical not in self._canonical_refs:
            return canonical, None, None, None
        ref_low, ref_high = self._canonical_refs[canonical]
        if ref_low == 0.0 and ref_high == 0.0:
            return canonical, "NO_REF", ref_low, ref_high
        if value < ref_low:
            classification = "LOW"
        elif value > ref_high:
            classification = "HIGH"
        else:
            classification = "NORMAL"
        return canonical, classification, ref_low, ref_high

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path
