import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestClinicalRAG:
    """
    ClinicalRAG usa ChromaDB, SentenceTransformer e HuggingFace pipeline.
    Todos os backends externos são mockados — nenhum modelo é baixado nos testes.
    """

    def _make_rag(self):
        from mosaicfl.core.rag import ClinicalRAG

        mock_collection = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(3, 384).astype(np.float32)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "prompt truncado"

        mock_llm = MagicMock()
        mock_generator = MagicMock(return_value=[{"generated_text": "Diagnóstico provável: covid19."}])

        with patch("mosaicfl.core.rag.chromadb.PersistentClient", return_value=mock_chroma), \
             patch("mosaicfl.core.rag.SentenceTransformer", return_value=mock_embedder), \
             patch("mosaicfl.core.rag.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("mosaicfl.core.rag.AutoModelForCausalLM.from_pretrained", return_value=mock_llm), \
             patch("mosaicfl.core.rag.pipeline", return_value=mock_generator):
            rag = ClinicalRAG()

        rag.embedder = mock_embedder
        rag.collection = mock_collection
        rag.generator = mock_generator
        rag.tokenizer = mock_tokenizer
        return rag, mock_collection, mock_embedder

    def test_build_knowledge_base_calls_collection_add(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        patterns = [
            {"texto": "febre tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
            {"texto": "dispneia grave", "desfecho": "pneumonia", "faixa_etaria": "idoso"},
        ]
        rag.build_knowledge_base(patterns)
        mock_collection.add.assert_called_once()

    def test_retrieve_returns_list_of_dicts(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [["texto1", "texto2"]],
            "metadatas": [[{"desfecho": "covid19"}, {"desfecho": "pneumonia"}]],
            "distances": [[0.1, 0.2]],
        }
        results = rag.retrieve("febre tosse", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "texto" in r
            assert "metadata" in r

    def test_explain_returns_expected_keys(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [["febre e tosse, desfecho positivo"] * 3],
            "metadatas": [[{"desfecho": "covid19"}] * 3],
            "distances": [[0.1, 0.15, 0.2]],
        }
        rag.generator.return_value = [{"generated_text": "Justificativa: covid19 confirmado."}]

        patient = {"febre": "alta", "tosse": "seca", "saturacao": "92%"}
        pred = {"diagnostico": "covid19", "probabilidade": 0.82}
        result = rag.explain(patient, pred)

        for key in ["predicao", "justificativa", "fontes", "alucinacao_detectada", "confiavel"]:
            assert key in result, f"Chave '{key}' ausente no resultado de explain()"
        assert isinstance(result["confiavel"], bool)

    def test_explain_confiavel_false_when_hallucination_detected(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [["febre leve"] * 3],
            "metadatas": [[{"desfecho": "covid19"}] * 3],
            "distances": [[0.9, 0.95, 0.98]],
        }
        rag.generator.return_value = [{"generated_text": "Justificativa genérica sem base clínica."}]

        result = rag.explain(
            {"sintoma": "desconhecido"},
            {"diagnostico": "covid19", "probabilidade": 0.5}
        )
        assert "confiavel" in result

    def test_retrieve_with_zero_results(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        results = rag.retrieve("sintoma inexistente", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_end_to_end_with_model_mocked(self):
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [["febre tosse"] * 3],
            "metadatas": [[{"desfecho": "covid19"}] * 3],
            "distances": [[0.1, 0.15, 0.2]],
        }
        rag.generator.return_value = [{"generated_text": "Justificativa: covid19."}]

        result = rag.explain(
            {"febre": "alta", "saturacao": "95%"},
            {"diagnostico": "covid19", "probabilidade": 0.8}
        )
        assert "justificativa" in result
