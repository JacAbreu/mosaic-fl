"""
Sistema RAG: construção de base de conhecimento + recuperação + geração.
Usa pgvector (knowledge.clinical_profiles) + all-MiniLM-L6-v2 + DistilGPT-2.
Quando FL_DB_URL não está configurado, usa _InMemoryStore (numpy) para experimentos.
"""
import os

import sqlalchemy as sa
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np
from typing import List, Dict, Tuple

from .config import FED_CFG, RUNTIME_CFG

_DEFAULT_DB_URL = os.getenv("FL_DB_URL", "")


# ---------------------------------------------------------------------------
# Store — interface ChromaDB-compatível sobre knowledge.clinical_profiles
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ClinicalRAG
# ---------------------------------------------------------------------------

class ClinicalRAG:
    def __init__(self, db_url: str = _DEFAULT_DB_URL) -> None:
        if db_url:
            engine = sa.create_engine(db_url, pool_pre_ping=True)
            self.collection = _PostgreSQLStore(engine)
        else:
            self.collection = _InMemoryStore()

        self.embedder = SentenceTransformer(RUNTIME_CFG.embedding_model)

        self.tokenizer = AutoTokenizer.from_pretrained(RUNTIME_CFG.llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(RUNTIME_CFG.llm_model)
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            max_new_tokens=FED_CFG.max_new_tokens,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_knowledge_base(self, patterns: List[Dict]) -> None:
        """
        Constrói base de conhecimento com perfis prototípicos anonimizados.
        patterns: lista de dicts com 'texto', 'desfecho', 'faixa_etaria'
        """
        texts, metadatas, ids = [], [], []
        for i, p in enumerate(patterns):
            anon_text = p["texto"].replace(
                str(p.get("idade_exacta", "")), p.get("faixa_etaria", "adulto")
            )
            texts.append(anon_text)
            metadatas.append({
                "desfecho":    p["desfecho"],
                "faixa_etaria": p.get("faixa_etaria", "adulto"),
                "categoria":   p.get("categoria", "geral"),
            })
            ids.append(f"profile_{i}")

        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query_text: str, top_k: int = FED_CFG.top_k) -> List[Dict]:
        """Recupera top-k perfis mais similares via distância de cosseno."""
        query_emb = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        docs = results["documents"][0]
        return [
            {
                "texto":    docs[i],
                "metadata": results["metadatas"][0][i],
                "distancia": results["distances"][0][i],
            }
            for i in range(len(docs))
        ]

    def generate_justification(
        self,
        prediction: str,
        probability: float,
        symptoms: str,
        retrieved_cases: List[Dict],
    ) -> Tuple:
        MAX_CASE_TOKENS = 100
        truncated_cases = []
        for c in retrieved_cases[:3]:
            ids = self.tokenizer.encode(c["texto"], max_length=MAX_CASE_TOKENS, truncation=True)
            truncated_cases.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        cases_text = "\n".join([f"- {t}" for t in truncated_cases])

        symptoms_ids = self.tokenizer.encode(symptoms, max_length=50, truncation=True)
        symptoms_trunc = self.tokenizer.decode(symptoms_ids, skip_special_tokens=True)

        prompt = (
            f"Com base nos seguintes casos clínicos semelhantes:\n{cases_text}\n\n"
            f"O modelo previu {prediction} (probabilidade {probability:.2f}) "
            f"para o paciente com sintomas: {symptoms_trunc}.\n"
            f"Justifique brevemente a predição:"
        )
        max_prompt_tokens = 1024 - FED_CFG.max_new_tokens - 10
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_prompt_tokens, truncation=True)
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        output = self.generator(prompt, do_sample=True, temperature=0.7, num_return_sequences=1)
        justification = output[0]["generated_text"].replace(prompt, "").strip()

        hallucination = probability < 0.6 and "certeza" in justification.lower()
        return justification, retrieved_cases, hallucination

    def explain(self, patient_data: Dict, model_prediction: Dict) -> Dict:
        # Modo banco: patient_data["tokens"] contém os tokens clínicos reais do paciente.
        # Modo CSV: usa campos demográficos/sintomáticos clássicos.
        if "tokens" in patient_data:
            query = patient_data["tokens"]
        else:
            query = (
                f"febre {patient_data.get('febre', '')}, "
                f"tosse {patient_data.get('tosse', '')}, "
                f"saturação {patient_data.get('saturacao', '')}, "
                f"idade {patient_data.get('faixa_etaria', '')}"
            )
        cases = self.retrieve(query)
        justification, sources, hallucinated = self.generate_justification(
            model_prediction["diagnostico"],
            model_prediction["probabilidade"],
            query,
            cases,
        )
        return {
            "predicao":            model_prediction,
            "justificativa":       justification,
            "fontes":              sources,
            "alucinacao_detectada": hallucinated,
            "confiavel":           not hallucinated and len(sources) > 0,
        }
