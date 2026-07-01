"""
stores.py — Backends de armazenamento vetorial (interface ChromaDB-compatível).

_InMemoryStore  — sem PostgreSQL/pgvector, similaridade de cosseno via numpy.
_PostgreSQLStore — knowledge.clinical_profiles (pgvector).
"""
from typing import Dict, List

import numpy as np
import sqlalchemy as sa
from sqlalchemy import text


class _InMemoryStore:
    """
    Store em memória para experimentos sem PostgreSQL/pgvector.
    Usa similaridade de cosseno via numpy — sem dependências externas.
    Interface idêntica à de _PostgreSQLStore e ChromaDB.
    """

    def __init__(self) -> None:
        self._embeddings: list = []
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []

    def add(
        self,
        embeddings: List,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        for emb, doc, meta in zip(embeddings, documents, metadatas):
            self._embeddings.append(np.array(emb, dtype=np.float32))
            self._documents.append(doc)
            self._metadatas.append(meta)

    def query(self, query_embeddings: List, n_results: int) -> Dict:
        if not self._embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)
        stored = np.stack(self._embeddings)                      # (N, dim)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        norms = np.linalg.norm(stored, axis=1, keepdims=True) + 1e-8
        sims = stored / norms @ q_norm                           # (N,)

        n = min(n_results, len(self._documents))
        top_idx = np.argsort(-sims)[:n]

        return {
            "documents": [[self._documents[i] for i in top_idx]],
            "metadatas": [[self._metadatas[i] for i in top_idx]],
            "distances": [[float(1.0 - sims[i]) for i in top_idx]],
        }


class _PostgreSQLStore:
    """
    Armazenamento de embeddings clínicos em knowledge.clinical_profiles.
    Interface propositalmente idêntica à da collection ChromaDB para que
    o ClinicalRAG não precise distinguir o backend nos seus métodos.
    """

    def __init__(self, engine: sa.Engine) -> None:
        self._engine = engine

    def add(
        self,
        embeddings: List,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        rows = [
            {
                "id":          id_,
                "document":    doc,
                "embedding":   "[" + ",".join(str(float(x)) for x in emb) + "]",
                "desfecho":    meta.get("desfecho"),
                "faixa_etaria": meta.get("faixa_etaria"),
                "categoria":   meta.get("categoria"),
            }
            for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings)
        ]
        with self._engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO knowledge.clinical_profiles
                    (id, document, embedding, desfecho, faixa_etaria, categoria)
                VALUES
                    (:id, :document, CAST(:embedding AS vector),
                     :desfecho, :faixa_etaria, :categoria)
                ON CONFLICT (id) DO UPDATE SET
                    document     = EXCLUDED.document,
                    embedding    = EXCLUDED.embedding,
                    desfecho     = EXCLUDED.desfecho,
                    faixa_etaria = EXCLUDED.faixa_etaria,
                    categoria    = EXCLUDED.categoria
            """), rows)

    def query(self, query_embeddings: List, n_results: int) -> Dict:
        emb_str = "[" + ",".join(str(float(x)) for x in query_embeddings[0]) + "]"
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT document, desfecho, faixa_etaria, categoria,
                       embedding <=> CAST(:emb AS vector) AS distance
                FROM knowledge.clinical_profiles
                ORDER BY distance
                LIMIT :n
            """), {"emb": emb_str, "n": n_results}).mappings().all()
        return {
            "documents": [[r["document"] for r in rows]],
            "metadatas": [[{
                "desfecho":    r["desfecho"],
                "faixa_etaria": r["faixa_etaria"],
                "categoria":   r["categoria"],
            } for r in rows]],
            "distances": [[r["distance"] for r in rows]],
        }
