"""engine.py — InferenceEngine: wrapper do modelo MOSAIC-FL para inferência single-patient em tempo real.

Substituição do MD5: a tokenização por hash gerava tokens fora do vocabulário
treinado, tornando o predict() clinicamente inválido.
"""
import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .compat import MAX_SEQ_LEN as _MAX_SEQ_LEN
from .compat import _DEFAULT_MC_SAMPLES, _MOSAICFL_AVAILABLE
from .tokenization import _load_alias_cache, _load_canonical_refs, _resolve_canonical, records_to_tokens

if _MOSAICFL_AVAILABLE:
    from mosaicfl.core.calibration import IsotonicCalibrator
    from mosaicfl.core.config import MODEL_CFG
    from mosaicfl.core.model import SimplifiedBEHRT

logger = logging.getLogger(__name__)


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

        self._vocab:             dict[str, int]             = {}
        self._alias_cache:       dict[str, str]             = {}
        self._canonical_refs:    dict[str, tuple[float, float]] = {}
        self._temperature:       float                      = 1.0
        self._calibration_method: str                       = "temperature"
        self._isotonic:          Optional["IsotonicCalibrator"] = None
        self._checkpoint_round:  Optional[int]              = None
        self._checkpoint_at:     Optional[str]              = None
        self._model_version:     Optional[str]              = None
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
        # weights_only=False: checkpoint contém vocab (dict str→int), temperatura, metadados
        # e, quando calibration_method="isotonic", objetos sklearn.IsotonicRegression (picklable,
        # não tensors) — mesmo trade-off já aceito em checkpoint_store/serialization._deserialize()
        # (arquivo local produzido pelo próprio pipeline de treino, não upload externo).
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "vocab" in state:
            self._vocab = state["vocab"]
            self.model.load_state_dict(state["model_state"])
            self._temperature      = float(state.get("temperature", 1.0))
            self._checkpoint_round = state.get("checkpoint_round")
            self._checkpoint_at    = state.get("checkpoint_at")
            self._model_version    = state.get("model_version")
            self._load_calibration_state(state)
        else:
            # Compatibilidade com checkpoints antigos (só pesos)
            self.model.load_state_dict(state)
            self._temperature = 1.0
            logger.warning(
                "inference_engine_legacy_checkpoint — vocabulário ausente; "
                "salve o vocab junto com os pesos do modelo"
            )
        self._checkpoint_path = path
        logger.info(
            "inference_engine_loaded path=%s vocab_size=%d method=%s T=%.4f",
            path, len(self._vocab), self._calibration_method, self._temperature,
        )

    def _load_calibration_state(self, state: dict) -> None:
        """Reconstrói o calibrador ativo a partir do checkpoint (temperature ou isotonic).

        Checkpoints salvos antes desta funcionalidade não têm "calibration_method" —
        tratados como "temperature" (comportamento prévio, sem quebra de compatibilidade)."""
        self._calibration_method = state.get("calibration_method", "temperature")
        self._isotonic = None
        if self._calibration_method == "isotonic":
            calibrators = state.get("isotonic_calibrators")
            num_classes = state.get("isotonic_num_classes", 0)
            if calibrators:
                self._isotonic = IsotonicCalibrator.from_calibrators(calibrators, num_classes)
            else:
                logger.warning(
                    "inference_engine_isotonic_missing — calibration_method=isotonic mas "
                    "isotonic_calibrators ausente/vazio; usando probabilidades não calibradas"
                )
                self._calibration_method = "temperature"

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

    def load_from_store(self, checkpoint: dict) -> None:
        """Carrega pesos e vocabulário a partir do dict retornado pelo CheckpointStore.

        Usado quando não há checkpoint em arquivo — o training pipeline salva no banco
        (SQLite ou PostgreSQL) e a API carrega via CheckpointStore.load_best().
        """
        if "vocab" not in checkpoint or "model_state" not in checkpoint:
            raise ValueError("checkpoint inválido: faltam campos 'vocab' e/ou 'model_state'")
        self._vocab            = checkpoint["vocab"]
        self._temperature      = float(checkpoint.get("temperature", 1.0))
        self._checkpoint_round = checkpoint.get("checkpoint_round")
        self._checkpoint_at    = checkpoint.get("checkpoint_at")
        self._model_version    = checkpoint.get("model_version")
        self.model.load_state_dict(checkpoint["model_state"])
        self._load_calibration_state(checkpoint)
        self._checkpoint_path  = Path("<checkpoint_store>")
        logger.info(
            "inference_engine_loaded_from_store vocab_size=%d method=%s T=%.4f round=%s",
            len(self._vocab), self._calibration_method, self._temperature, self._checkpoint_round,
        )

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

    def predict_proba(self, exam_records: list, mc_samples: int = _DEFAULT_MC_SAMPLES) -> dict:
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
                "probabilities":    {l: dict(empty) for l in labels},
                "predicted_class":  0,
                "predicted_label":  labels[0],
                "mc_samples":       0,
                "risk_score":       0.0,
                "temperature":      self._temperature,
                "trained":          self._checkpoint_path is not None,
                "calibrated":       self._is_calibrated(),
                "checkpoint_round": self._checkpoint_round,
                "checkpoint_at":    self._checkpoint_at,
                "model_version":    self._model_version,
            }

        x, mask = self._tokenize(exam_records)
        use_isotonic = self._calibration_method == "isotonic" and self._isotonic is not None

        # Lock garante que model.train()/eval() não colidem entre requests simultâneos
        with self._mc_lock:
            self.model.train()
            all_probs: list = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    if use_isotonic:
                        logits = self.model(x, mask=mask)
                        probs  = self._isotonic.calibrate_probs(F.softmax(logits, dim=-1))
                    else:
                        logits = self.model(x, mask=mask) / max(self._temperature, 1e-3)
                        probs  = F.softmax(logits, dim=-1)
                    all_probs.append(probs[0])
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
            "predicted_class":  predicted_class,
            "predicted_label":  labels[predicted_class],
            "mc_samples":       mc_samples,
            "risk_score":       round(risk_score, 4),
            "temperature":      self._temperature,
            "trained":          self._checkpoint_path is not None,
            "calibrated":       self._is_calibrated(),
            "calibration_method": self._calibration_method,
            "checkpoint_round": self._checkpoint_round,
            "checkpoint_at":    self._checkpoint_at,
            "model_version":    self._model_version,
        }

    def _is_calibrated(self) -> bool:
        """True se algum calibrador estiver ativo (temperatura ≠ 1.0 ou isotônica ajustada)."""
        if self._calibration_method == "isotonic":
            return self._isotonic is not None
        return self._temperature != 1.0

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
