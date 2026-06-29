"""Classe FederatedTraining — orquestra carregamento, FL, RAG, baseline e ablation."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import DataLoader

from infrastructure.shared.metrics_store import get_metrics_store
from mosaicfl.core.config import FL_DB_URL, MODEL_CFG
from mosaicfl.core.data_loader import load_with_fallback
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.preprocessor import EHRPreprocessor

from .ablation import run_ablation_demographics, run_pooled_behrt
from .baselines import run_baseline_rf
from .dataloaders import prepare_dataloaders, prepare_dataloaders_from_db
from .fl_core import run_federated_learning
from .rag import run_rag_pipeline

logger = logging.getLogger(__name__)


class FederatedTraining:
    """
    Encapsula o pipeline MOSAIC-FL completo: carregamento, FL, RAG, baseline e ablation.

    Usada pelos dois orquestradores:
      - run_training.py              → instancia com db_url real (dados FAPESP)
      - run_experiments_simulation.py → instancia sem db_url (dados sintéticos)
    """

    def __init__(
        self,
        log_file: str,
        db_url: Optional[str] = None,
        data_source: str = "synthetic",
    ) -> None:
        self.log_file    = log_file
        self.db_url      = db_url
        self.data_source = data_source

        self.client_loaders:        Optional[Dict]       = None
        self.test_loader:           Optional[DataLoader] = None
        self.vocab_map:             Optional[Dict]       = None
        self.total:                 int                  = 0
        self.demographics_by_client: Optional[Dict]      = None
        self.test_loader_demo:      Optional[DataLoader] = None
        self.cal_loader:            Optional[DataLoader] = None
        self.history:               Optional[Dict]       = None
        self.global_model:          Optional[SimplifiedBEHRT] = None
        self._last_round:           int                  = 0

        self._metrics_store = get_metrics_store(db_url)

    def load_from_db(self, db_url: str) -> None:
        (
            self.client_loaders,
            self.test_loader,
            self.vocab_map,
            self.total,
            self.demographics_by_client,
            self.test_loader_demo,
            self.cal_loader,
        ) = prepare_dataloaders_from_db(db_url)

    def load_synthetic(self) -> None:
        df_raw = load_with_fallback(allow_synthetic=True)
        preprocessor = EHRPreprocessor()
        self.client_loaders, self.test_loader, self.vocab_map, self.total = \
            prepare_dataloaders(df_raw, preprocessor)

    def train(self) -> None:
        self.history, self.global_model = run_federated_learning(
            self.client_loaders,
            self.test_loader,
            self.total,
            vocab=self.vocab_map,
            cal_loader=self.cal_loader,
        )
        if self.history and "rounds" in self.history and self.history["rounds"]:
            self._last_round = self.history["rounds"][-1]
            self._metrics_store.save(
                round_num=self._last_round,
                metrics={
                    "accuracy":      self.history["accuracy"][-1] if self.history.get("accuracy") else None,
                    "loss":          self.history["loss"][-1] if self.history.get("loss") else None,
                    "macro_auc":     None,
                    "macro_f1":      None,
                    "ece":           None,
                    "per_class_auc": None,
                    "per_class_f1":  None,
                },
                data_source=self.data_source,
            )

    def run_rag(self) -> Dict:
        result = run_rag_pipeline(self.global_model, self.vocab_map, self.test_loader)
        p_metrics = result.get("precision_metrics", {})
        if p_metrics:
            k = p_metrics.get("k", 3)
            self._metrics_store.save(
                round_num=self._last_round,
                metrics={
                    "rag_precision_at_k":      p_metrics.get(f"precision_at_{k}"),
                    "rag_k":                   k,
                    "rag_per_class_precision": p_metrics.get(f"per_class_precision_at_{k}"),
                },
                data_source=self.data_source,
            )
        return result

    def run_baseline(self) -> Dict:
        result = run_baseline_rf(
            self.client_loaders,
            self.test_loader,
            class_labels=list(MODEL_CFG.class_labels),
        )
        baseline_path = (
            Path("experiments/data")
            / f"baseline_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Baseline salvo: {baseline_path}")
        m_a = result.get("opcao_a_centralizado") or {}
        if m_a:
            self._metrics_store.save(
                round_num=0,
                metrics={
                    "accuracy":  m_a.get("accuracy"),
                    "macro_auc": m_a.get("macro_auc"),
                    "macro_f1":  m_a.get("macro_f1"),
                    "ece":       m_a.get("ece"),
                },
                data_source=f"{self.data_source}_baseline_rf",
            )
        return result

    def run_ablation(self) -> Dict:
        result = run_ablation_demographics(
            client_loaders=self.client_loaders,
            test_loader=self.test_loader,
            demographics_by_client=self.demographics_by_client,
            test_loader_demo=self.test_loader_demo,
            n_epochs=10,
            seeds=[42, 7, 123],
        )
        ablation_path = (
            Path("experiments/data")
            / f"ablation_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Ablation salvo: {ablation_path}")
        return result

    def summarize(self, rag_result: Dict, baseline_result: Dict, ablation_result: Dict) -> None:
        logger.info("=" * 60)
        logger.info("CONCLUÍDO")
        logger.info(f"  Modo dados:     {self.data_source}")
        logger.info(f"  Clientes FL:    {len(self.client_loaders)}")
        logger.info(f"  RAG confiável:  {rag_result.get('confiavel', False)}")
        m_a = baseline_result.get("opcao_a_centralizado") or {}
        if m_a:
            logger.info(
                f"  RF centralizado: Acc={m_a.get('accuracy','?')}  "
                f"AUC={m_a.get('macro_auc','?')}  F1={m_a.get('macro_f1','?')}"
            )
        delta = ablation_result.get("delta_B_minus_A", {})
        if delta:
            logger.info(f"  Ablation Δ Acc: {delta.get('accuracy', 'n/a'):+}")
        logger.info(f"  Logs em:        {self.log_file}")
        logger.info("=" * 60)

        logger.info(
            "TREINAMENTO_COMPLETO status=ok fl_rounds=%d rag_ok=%s baseline_rf_ok=%s ablation_ok=%s log=%s",
            self._last_round,
            not bool(rag_result.get("erro")),
            not bool(baseline_result.get("erro")),
            not bool(ablation_result.get("erro")),
            self.log_file,
        )
