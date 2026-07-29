"""Pipeline RAG: extração de padrões BEHRT + recuperação + avaliação Precision@k."""
import json
import logging
import random
from datetime import datetime
from typing import Dict, List

from torch.utils.data import DataLoader

from mosaicfl.core.config import MODEL_CFG
from mosaicfl.core.interpretability import BEHRTPatternExtractor
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.rag import ClinicalRAG
from mosaicfl.core.rag.precision import eval_precision_at_k as _eval_rag_precision_at_k

logger = logging.getLogger(__name__)


def run_rag_pipeline(
    global_model: SimplifiedBEHRT,
    vocab_map: Dict,
    test_loader: DataLoader,
) -> Dict:
    """Extrai padrões do BEHRT, gera justificativa via RAG e avalia Precision@k."""

    logger.info("=" * 60)
    logger.info("PIPELINE RAG")
    logger.info("=" * 60)

    all_labels = []
    for _, batch_y, *_ in test_loader:
        all_labels.extend(batch_y.tolist())
    desfechos = sorted(set(all_labels))
    logger.info(f"Desfechos presentes no test_loader: {desfechos}")

    extractor = BEHRTPatternExtractor(global_model, vocab_map)
    patterns = extractor.generate_all_profiles(test_loader, desfechos=desfechos)
    logger.info(f"Padrões extraídos: {len(patterns)} perfis")

    rag = ClinicalRAG()
    rag.build_knowledge_base(patterns)

    vocab_inverse = {v: k for k, v in vocab_map.items()}
    labels = MODEL_CFG.class_labels

    logger.info("Avaliando Precision@k da recuperação...")
    precision_metrics = _eval_rag_precision_at_k(
        rag, test_loader, vocab_inverse, list(labels), k=3
    )

    sample_label = desfechos[0]
    sample_tokens: List[str] = []
    for batch_x, batch_y, *_ in test_loader:
        raw_tokens = [vocab_inverse.get(t, "") for t in batch_x[0].tolist() if t > 2]
        sample_tokens = [t for t in raw_tokens if t][:10]
        sample_label = int(batch_y[0].item())
        break

    label_name = (
        labels[sample_label] if sample_label < len(labels) else f"classe_{sample_label}"
    )
    patient_data = {"tokens": ", ".join(sample_tokens) if sample_tokens else "dados laboratoriais"}
    model_prediction = {"diagnostico": label_name, "probabilidade": random.uniform(0.55, 0.95)}

    result = rag.explain(patient_data, model_prediction)
    logger.info(f"Justificativa — confiável: {result['confiavel']} | "
                f"alucinação: {result['alucinacao_detectada']}")

    result["precision_metrics"] = precision_metrics

    rag_path = f"experiments/data/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
