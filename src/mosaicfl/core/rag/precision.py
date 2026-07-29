"""precision.py — Avaliação de Precision@k da recuperação do RAG.

Compartilhado entre Caminho A (experiments/training/core/rag.py, test_loader
centralizado — só possível em simulação) e Caminho B (src/mosaicfl/core/client.py,
FedProxClient.evaluate() usa self.val_loader LOCAL — nunca dado bruto de outro
hospital). Extraído pra cá em 2026-07-28 pra não duplicar a lógica entre os dois
caminhos (mesma decisão já tomada pra BEHRTPatternExtractor).
"""
import logging
from typing import Dict, List

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def eval_precision_at_k(
    rag,
    loader: DataLoader,
    vocab_inverse: Dict[int, str],
    class_labels: List[str],
    k: int = 3,
) -> Dict:
    """
    Avalia a qualidade da recuperação do RAG via Precision@k.

    Para cada amostra do loader, consulta o RAG com os tokens do paciente e
    verifica quantos dos k casos recuperados têm o mesmo desfecho que o
    rótulo real. Métrica central para CDSS humano-no-loop — mede se o que é
    RECUPERADO é relevante, não se a justificativa GERADA é boa (isso é a
    avaliação Likert, um problema diferente).
    """
    hits_total = 0
    queries_total = 0
    per_class_hits: Dict[str, int] = {lbl: 0 for lbl in class_labels}
    per_class_queries: Dict[str, int] = {lbl: 0 for lbl in class_labels}

    for batch_x, batch_y, *_ in loader:
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
        "n_queries": queries_total // k if k > 0 else 0,
    }
