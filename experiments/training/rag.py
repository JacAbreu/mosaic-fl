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

logger = logging.getLogger(__name__)


def _eval_rag_precision_at_k(
    rag: ClinicalRAG,
    test_loader: DataLoader,
    vocab_inverse: Dict[int, str],
    class_labels: List[str],
    k: int = 3,
) -> Dict:
    """
    Avalia a qualidade da recuperação do RAG via Precision@k.

    Para cada amostra do test_loader, consulta o RAG com os tokens do paciente
    e verifica quantos dos k casos recuperados têm o mesmo desfecho que o rótulo
    real. Métrica central para CDSS humano-no-loop.
    """
    hits_total = 0
    queries_total = 0
    per_class_hits: Dict[str, int] = {lbl: 0 for lbl in class_labels}
    per_class_queries: Dict[str, int] = {lbl: 0 for lbl in class_labels}

    for batch_x, batch_y in test_loader:
        for seq, label_idx in zip(batch_x.tolist(), batch_y.tolist()):
            tokens = [vocab_inverse[t] for t in seq if t > 2 and t in vocab_inverse]
            if not tokens:
                continue

            ground_truth = (
                class_labels[label_idx] if label_idx < len(class_labels)
                else f"classe_{label_idx}"
            )
            query = ", ".join(tokens[:20])
            retrieved = rag.retrieve(query, top_k=k)

            n_hits = sum(
                1 for c in retrieved
                if c.get("metadata", {}).get("desfecho") == ground_truth
            )
            hits_total += n_hits
            queries_total += k
            per_class_hits[ground_truth] = per_class_hits.get(ground_truth, 0) + n_hits
            per_class_queries[ground_truth] = per_class_queries.get(ground_truth, 0) + k

    precision_at_k = round(hits_total / queries_total, 4) if queries_total > 0 else 0.0
    per_class_precision = {
        lbl: round(per_class_hits[lbl] / per_class_queries[lbl], 4)
        if per_class_queries[lbl] > 0 else None
        for lbl in class_labels
    }

    logger.info(f"RAG Precision@{k} (recuperação): {precision_at_k:.4f}")
    for lbl, p in per_class_precision.items():
        logger.info(f"  {lbl}: {p:.4f}" if p is not None else f"  {lbl}: n/a")

    return {
        f"precision_at_{k}": precision_at_k,
        f"per_class_precision_at_{k}": per_class_precision,
        "k": k,
        "n_queries": queries_total // k,
    }


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
    for _, batch_y in test_loader:
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
    for batch_x, batch_y in test_loader:
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
