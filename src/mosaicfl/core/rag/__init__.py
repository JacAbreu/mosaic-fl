"""
Sistema RAG: construção de base de conhecimento + recuperação + geração.
Usa pgvector (knowledge.clinical_profiles) + all-MiniLM-L6-v2 + LLM configurável.
Backend LLM selecionado por FL_LLM_BACKEND:
  huggingface  → AutoModelForCausalLM (padrão: distilgpt2)
  ollama       → POST localhost:11434/api/generate (ex: gemma3:4b)
Quando FL_DB_URL não está configurado, usa _InMemoryStore (numpy) para experimentos.

Submódulos:
  stores.py        — _InMemoryStore, _PostgreSQLStore (armazenamento vetorial)
  llm_backends.py  — _check_ollama_available, _generate_ollama, _load_huggingface_backend

ClinicalRAG permanece neste arquivo (e não em um submódulo próprio) porque os testes
fazem patch direto em "mosaicfl.core.rag.sa", ".SentenceTransformer",
"._check_ollama_available" e "._load_huggingface_backend" — mock só intercepta a
chamada quando o nome é importado no mesmo módulo que o utiliza.
"""
import logging
import os
from typing import Dict, List, Tuple

import sqlalchemy as sa
from sentence_transformers import SentenceTransformer

from ..config import FED_CFG, LLM_BACKEND, MAX_SEQ_LEN, RUNTIME_CFG, FL_ENV
from .llm_backends import _check_ollama_available, _generate_ollama, _load_huggingface_backend
from .stores import _InMemoryStore, _PostgreSQLStore

logger = logging.getLogger(__name__)

_DEFAULT_DB_URL = os.getenv("FL_DB_URL", "")


class ClinicalRAG:
    def __init__(self, db_url: str = _DEFAULT_DB_URL) -> None:
        if db_url:
            engine = sa.create_engine(db_url, pool_pre_ping=True)
            self.collection = _PostgreSQLStore(engine)
        else:
            self.collection = _InMemoryStore()

        self.embedder = SentenceTransformer(RUNTIME_CFG.embedding_model)

        self._llm_model       = RUNTIME_CFG.llm_model
        self._llm_was_fallback = False

        requested_backend = LLM_BACKEND
        if requested_backend == "ollama" and not _check_ollama_available():
            hf_fallback = RUNTIME_CFG.llm_hf_model
            self._llm_was_fallback = True
            _msg = (
                "RAG: Ollama solicitado (modelo: %s) mas não está acessível em "
                "localhost:11434. Usando HuggingFace (%s) como fallback. "
                "Resultados desta run NÃO são comparáveis com runs usando Ollama. "
                "Para usar Ollama: 'ollama serve' e 'ollama pull %s'."
            )
            if FL_ENV == "test":
                logger.warning(_msg, self._llm_model, hf_fallback, self._llm_model)
            else:
                logger.error(_msg, self._llm_model, hf_fallback, self._llm_model)
            requested_backend = "huggingface"
            self._llm_model = hf_fallback

        self._llm_backend = requested_backend

        if self._llm_backend == "ollama":
            self.tokenizer = None
            self.generator = None
            logger.info("RAG LLM backend: ollama | model: %s", self._llm_model)
        else:
            self.tokenizer, self.generator = _load_huggingface_backend(self._llm_model)
            logger.info("RAG LLM backend: huggingface | model: %s", self._llm_model)

    def build_knowledge_base(self, patterns: List[Dict]) -> None:
        """
        Constrói base de conhecimento com perfis prototípicos anonimizados.
        patterns: lista de dicts com 'texto', 'desfecho', 'faixa_etaria'
        """
        texts, metadatas, ids = [], [], []
        for i, p in enumerate(patterns):
            idade_exacta = str(p.get("idade_exacta", ""))
            if idade_exacta:
                anon_text = p["texto"].replace(idade_exacta, p.get("faixa_etaria", "adulto"))
            else:
                anon_text = p["texto"]
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
        if self._llm_backend == "ollama":
            # Truncagem por caracteres (~4 chars/token) — sem tokenizador.
            MAX_CASE_CHARS = 400
            truncated_cases = [c["texto"][:MAX_CASE_CHARS] for c in retrieved_cases[:3]]
            cases_text = "\n".join([f"- {t}" for t in truncated_cases])
            symptoms_trunc = symptoms[:MAX_SEQ_LEN * 4]

            prompt = (
                f"Você é um assistente clínico especializado. "
                f"Com base nos seguintes casos similares recuperados:\n{cases_text}\n\n"
                f"O modelo de IA previu o desfecho '{prediction}' "
                f"(probabilidade {probability:.2f}) para um paciente com os seguintes exames: {symptoms_trunc}.\n\n"
                f"Em 2 a 3 frases, justifique clinicamente esta predição com base nos casos similares:"
            )
            max_prompt_chars = (1024 - FED_CFG.max_new_tokens) * 4
            if len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars]
            justification = _generate_ollama(self._llm_model, prompt, FED_CFG.max_new_tokens)
        else:
            MAX_CASE_TOKENS = 100
            truncated_cases = []
            for c in retrieved_cases[:3]:
                ids = self.tokenizer.encode(c["texto"], max_length=MAX_CASE_TOKENS, truncation=True)
                truncated_cases.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            cases_text = "\n".join([f"- {t}" for t in truncated_cases])

            symptoms_ids = self.tokenizer.encode(symptoms, max_length=MAX_SEQ_LEN, truncation=True)
            symptoms_trunc = self.tokenizer.decode(symptoms_ids, skip_special_tokens=True)

            prompt = (
                f"Você é um assistente clínico especializado. "
                f"Com base nos seguintes casos similares recuperados:\n{cases_text}\n\n"
                f"O modelo de IA previu o desfecho '{prediction}' "
                f"(probabilidade {probability:.2f}) para um paciente com os seguintes exames: {symptoms_trunc}.\n\n"
                f"Em 2 a 3 frases, justifique clinicamente esta predição com base nos casos similares:"
            )
            max_prompt_tokens = 1024 - FED_CFG.max_new_tokens - 10
            prompt_ids = self.tokenizer.encode(prompt, max_length=max_prompt_tokens, truncation=True)
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            output = self.generator(
                prompt,
                max_new_tokens=FED_CFG.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
            )
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
            "predicao":             model_prediction,
            "justificativa":        justification,
            "fontes":               sources,
            "alucinacao_detectada": hallucinated,
            "confiavel":            not hallucinated and len(sources) > 0,
            "llm_backend":          self._llm_backend,
            "llm_model_used":       self._llm_model,
            "llm_was_fallback":     self._llm_was_fallback,
        }
