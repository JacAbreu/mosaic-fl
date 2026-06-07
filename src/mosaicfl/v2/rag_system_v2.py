"""
Sistema RAG: construção de base de conhecimento + recuperação + geração.
Usa ChromaDB + all-MiniLM-L6-v2 + DistilGPT-2.
"""
import chromadb
#from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np
from typing import List, Dict, Tuple
import json

from .config import FED_CFG, RUNTIME_CFG


class ClinicalRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(RUNTIME_CFG.embedding_model)
        #self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(RUNTIME_CFG.chroma_path)))
        self.chroma_client = chromadb.PersistentClient(path=str(RUNTIME_CFG.chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(name="clinical_profiles")

        # LLM leve: DistilGPT-2
        self.tokenizer = AutoTokenizer.from_pretrained(RUNTIME_CFG.llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(RUNTIME_CFG.llm_model)
        self.generator = pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer,
                                  max_new_tokens=FED_CFG.max_new_tokens, device=0 if torch.cuda.is_available() else -1,
                                  truncation=True)          # evita IndexError por prompt > 1024 tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_knowledge_base(self, patterns: List[Dict]):
        """
        Constrói base de conhecimento com perfis prototípicos anonimizados.
        patterns: lista de dicts com 'texto', 'desfecho', 'faixa_etaria'
        """
        texts = []
        metadatas = []
        ids = []
        
        for i, p in enumerate(patterns):
            # Anonimização estrutural
            anon_text = p['texto'].replace(str(p.get('idade_exacta', '')), p.get('faixa_etaria', 'adulto'))
            texts.append(anon_text)
            metadatas.append({
                "desfecho": p['desfecho'],
                "faixa_etaria": p.get('faixa_etaria', 'adulto'),
                "categoria": p.get('categoria', 'geral')
            })
            ids.append(f"profile_{i}")
        
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
        print(f"Base de conhecimento: {len(texts)} perfis inseridos.")

    def retrieve(self, query_text: str, top_k: int = FED_CFG.top_k) -> List[Dict]:
        """Recupera top-k documentos mais similares via distância de cosseno."""
        query_emb = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        
        docs = results['documents'][0]
        retrieved = []
        for i in range(len(docs)):
            retrieved.append({
                "texto": docs[i],
                "metadata": results['metadatas'][0][i],
                "distancia": results['distances'][0][i],
            })
        return retrieved

    def generate_justification(self, prediction: str, probability: float, 
                               symptoms: str, retrieved_cases: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Gera justificativa textual fundamentada nos casos recuperados.
        Retorna: (justificativa, casos_usados)
        """
        # Trunca cada caso a 100 tokens para não estourar o contexto do GPT-2 (limite: 1024)
        MAX_CASE_TOKENS = 100
        truncated_cases = []
        for c in retrieved_cases[:3]:
            ids = self.tokenizer.encode(c['texto'], max_length=MAX_CASE_TOKENS, truncation=True)
            truncated_cases.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        cases_text = "\n".join([f"- {t}" for t in truncated_cases])

        # Trunca os sintomas a 50 tokens
        symptoms_ids = self.tokenizer.encode(symptoms, max_length=50, truncation=True)
        symptoms_trunc = self.tokenizer.decode(symptoms_ids, skip_special_tokens=True)

        prompt = f"""Com base nos seguintes casos clínicos semelhantes:
{cases_text}

O modelo previu {prediction} (probabilidade {probability:.2f}) para o paciente com sintomas: {symptoms_trunc}.
Justifique brevemente a predição:"""

        # Garante que o prompt inteiro caiba no contexto (1024 - MAX_NEW_TOKENS)
        max_prompt_tokens = 1024 - FED_CFG.max_new_tokens - 10
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_prompt_tokens, truncation=True)
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        output = self.generator(prompt, do_sample=True, temperature=0.7, num_return_sequences=1)
        justification = output[0]['generated_text'].replace(prompt, "").strip()
        
        # Detecção simples de alucinação
        hallucination = False
        if probability < 0.6 and "certeza" in justification.lower():
            hallucination = True
        
        return justification, retrieved_cases, hallucination

    def explain(self, patient_data: Dict, model_prediction: Dict) -> Dict:
        """
        Pipeline completo: dado um paciente, retorna predição + justificativa.
        """
        query = f"febre {patient_data.get('febre', '')}, tosse {patient_data.get('tosse', '')}, " \
                f"saturação {patient_data.get('saturacao', '')}, idade {patient_data.get('faixa_etaria', '')}"
        
        cases = self.retrieve(query)
        justification, sources, hallucinated = self.generate_justification(
            model_prediction['diagnostico'],
            model_prediction['probabilidade'],
            query,
            cases
        )
        
        return {
            "predicao": model_prediction,
            "justificativa": justification,
            "fontes": sources,
            "alucinacao_detectada": hallucinated,
            "confiavel": not hallucinated and len(sources) > 0
        }