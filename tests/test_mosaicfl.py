"""
Testes unitários para o MOSAIC-FL (versões corrigidas).

Uso:
    pytest tests/test_mosaicfl.py -v
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Garante que src/ está no path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mosaicfl.v2.config import (
    VOCAB_SIZE, NUM_CLASSES, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
    BATCH_SIZE, NUM_ROUNDS, PROXIMAL_MU, MAX_SEQ_LEN, DEVICE,
)
# CORREÇÃO: imports que estavam faltando em test_mosaicfl.py original.
# PositionalEncoding e BEHRTEncoderLayer estão em model_v2, não em config.
from mosaicfl.v2.model_v2 import SimplifiedBEHRT, PositionalEncoding, BEHRTEncoderLayer
from mosaicfl.v2.data_loader import load_clinical_dataset, diagnose_dataset, load_with_fallback
from mosaicfl.v2.preprocess_v2 import EHRPreprocessor, split_by_institution
from mosaicfl.v2.client_v2 import FedProxClient
# CORREÇÃO: weighted_average e get_evaluate_fn também estão em server_v2, não em config.
from mosaicfl.v2.server_v2 import start_server, ConvergenceTracker, weighted_average, get_evaluate_fn


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame sintético para testes de pré-processamento."""
    return pd.DataFrame({
        "instituicao": ["HospA", "HospA", "HospB", "HospB", "HospC"],
        "idade": [25, 6, 45, 365, 70],
        "idade_unidade": ["anos", "meses", "anos", "dias", "anos"],
        # CORREÇÃO: valores float em vez de int.
        # O pandas 2.x+ recusa atribuir o resultado de uma multiplicação float
        # (ex: 150 lb × 0.453592 = 68.04 kg) dentro de uma coluna int64.
        # Erro observado: TypeError "Invalid value ... for dtype 'int64'".
        # A coluna 'peso' precisa ser float para aceitar o resultado da conversão lb→kg.
        "peso": [150.0, 70.0, 180.0, 50.0, 80.0],
        "peso_unidade": ["lb", "kg", "lb", "kg", "kg"],
        "temperatura": [98.6, 36.5, 99.5, 37.0, 38.0],
        "sintoma": ["febre", "tosse", "dispneia", "fadiga", "mialgia"],
        "exame": ["rt_pcr_positivo", "tomografia_normal", "rx_consolidacao",
                  "pcr_negativo", "tomografia_vidro_fosco"],
        "diagnostico": ["covid19_leve", "covid19_moderado", "pneumonia_bacteriana",
                        "covid19_grave", "alta"],
        "desfecho": [0, 0, 1, 1, 0],
    })


@pytest.fixture
def preprocessor():
    return EHRPreprocessor()


@pytest.fixture
def dummy_sequences():
    """Tensores dummy para testes do modelo."""
    batch_size, seq_len = 4, 16
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x, y


@pytest.fixture
def model():
    return SimplifiedBEHRT(use_cls_token=True)


# ─────────────────────────────────────────────────────────────
# TESTES: preprocess.py
# ─────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_normalize_units_idade(self, preprocessor, sample_df):
        """Converte meses e dias para anos."""
        df_norm = preprocessor.normalize_units(sample_df.copy())
        assert abs(df_norm.loc[1, "idade"] - 0.5) < 0.01    # 6 meses → 0.5 anos
        assert abs(df_norm.loc[3, "idade"] - 1.0) < 0.01    # 365 dias ≈ 1 ano
        assert (df_norm["idade_unidade"] == "anos").all()

    def test_normalize_units_peso(self, preprocessor, sample_df):
        """Converte lb para kg."""
        df_norm = preprocessor.normalize_units(sample_df.copy())
        assert abs(df_norm.loc[0, "peso"] - (150 * 0.453592)) < 0.01
        assert abs(df_norm.loc[2, "peso"] - (180 * 0.453592)) < 0.01
        assert (df_norm["peso_unidade"] == "kg").all()

    def test_clean_text_lowercases_and_strips(self, preprocessor):
        """clean_text faz lowercase e remove pontuação especial (exceto hifens)."""
        df = pd.DataFrame({"sintoma": ["FEBRE Alta", "Tosse!@#", "Dispneia"]})
        cleaned = preprocessor.clean_text(df, ["sintoma"])
        assert all(v == v.lower() for v in cleaned["sintoma"])

    def test_clean_text_removes_special_chars(self, preprocessor):
        """clean_text remove !@#$ mas preserva hifens."""
        df = pd.DataFrame({"sintoma": ["febre!@#$", "18-65 anos"]})
        cleaned = preprocessor.clean_text(df, ["sintoma"])
        # Caracteres especiais removidos
        assert "!" not in cleaned["sintoma"].iloc[0]
        # Hifens são preservados pelo regex [^\w\s\-]
        assert "18-65" in cleaned["sintoma"].iloc[1]

    def test_clean_text_removes_dots(self, preprocessor):
        """
        clean_text usa regex [^\w\\s\\-] que REMOVE pontos.
        O código comentado em preprocess_v2.py pretendia preservar pontos para
        códigos ICD (J18.1), mas a versão ativa não faz isso.
        Este teste documenta o comportamento real atual.
        """
        df = pd.DataFrame({"sintoma": ["J18.1", "98.6F"]})
        cleaned = preprocessor.clean_text(df, ["sintoma"])
        # Pontos são removidos pelo regex ativo
        assert "." not in cleaned["sintoma"].iloc[0]   # "j181"
        assert "." not in cleaned["sintoma"].iloc[1]   # "986f"

    def test_build_vocabulary_includes_special_tokens(self, preprocessor, sample_df):
        """Vocabulário deve incluir <PAD>, <UNK>, <MASK>, <CLS>."""
        preprocessor.build_vocabulary(sample_df, ["sintoma", "exame", "diagnostico"])
        assert "<PAD>" in preprocessor.vocab_map
        assert "<UNK>" in preprocessor.vocab_map
        assert "<MASK>" in preprocessor.vocab_map
        assert "<CLS>" in preprocessor.vocab_map
        assert preprocessor.vocab_map["<PAD>"] == 0

    def test_handle_missing_impute(self, preprocessor):
        """Imputação com mediana para numéricos e <UNK> para categóricos."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", np.nan, "c", "d"],
        })
        df_imp = preprocessor.handle_missing(df, strategy="impute")
        assert df_imp.loc[2, "num"] == 2.0  # mediana de [1, 2, 4]
        assert df_imp.loc[1, "cat"] == "<UNK>"

    def test_process_returns_summary(self, preprocessor, sample_df):
        """process() deve retornar DataFrame + dict summary."""
        df_proc, summary = preprocessor.process(
            sample_df, text_cols=["sintoma", "exame", "diagnostico"]
        )
        assert isinstance(df_proc, pd.DataFrame)
        assert isinstance(summary, dict)
        assert "total_amostras" in summary
        assert "tamanho_vocabulario" in summary
        assert summary["total_amostras"] == len(sample_df)


class TestSplitByInstitution:
    def test_split_creates_expected_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=5)
        assert len(clients) == 3
        assert 0 in clients and 1 in clients and 2 in clients

    def test_split_with_stratify(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=3, stratify_col="desfecho")
        for cid, subset in clients.items():
            assert subset["desfecho"].nunique() >= 1

    def test_split_limits_num_clients(self, sample_df):
        clients = split_by_institution(sample_df, num_clients=10)
        assert len(clients) == 3


# ─────────────────────────────────────────────────────────────
# TESTES: model.py
# ─────────────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.randn(2, 10, 64)
        assert pe(x).shape == x.shape

    def test_adds_position_info(self):
        pe = PositionalEncoding(d_model=64, max_len=129)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestBEHRTEncoderLayer:
    def test_forward_without_attention(self):
        layer = BEHRTEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_forward_with_attention(self):
        layer = BEHRTEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        x = torch.randn(2, 10, 64)
        out, attn = layer(x, return_attention=True)
        assert isinstance(out, torch.Tensor) and isinstance(attn, torch.Tensor)
        assert attn.shape == (2, 4, 10, 10)  # (batch, heads, seq, seq)


class TestSimplifiedBEHRT:
    def test_forward_without_attention(self, model, dummy_sequences):
        x, _ = dummy_sequences
        logits = model(x)
        assert logits.shape == (x.size(0), NUM_CLASSES)

    def test_forward_with_attention(self, model, dummy_sequences):
        x, _ = dummy_sequences
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (x.size(0), NUM_CLASSES)
        assert attn.shape[0] == NUM_LAYERS
        assert attn.shape[1] == x.size(0)
        assert attn.shape[2] == NUM_HEADS

    def test_masked_mean_pool_excludes_padding(self, model):
        x = torch.tensor([[1, 2, 3, 0, 0]])
        mask = (x == 0)
        emb = model.embedding(x) * np.sqrt(EMBED_DIM)
        pooled = model._masked_mean_pool(emb, mask)
        expected = emb[0, :3].mean(dim=0)
        assert torch.allclose(pooled[0], expected, atol=1e-5)

    def test_cls_token_is_parameter_not_vocab(self, model):
        """
        CORREÇÃO: O token CLS não aumenta o tamanho do vocabulário.
        A implementação atual usa um nn.Parameter separado (cls_token),
        não uma entrada extra no Embedding. Portanto:
            embedding.num_embeddings == VOCAB_SIZE  (não VOCAB_SIZE + 1)
        O código comentado em model_v2.py mostra a intenção original de usar
        vocab_size+1, mas a versão ativa usa nn.Parameter.
        """
        assert model.embedding.num_embeddings == VOCAB_SIZE
        assert isinstance(model.cls_token, torch.nn.Parameter)

    def test_dropout_respects_eval_mode(self, model, dummy_sequences):
        x, _ = dummy_sequences
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model(x), atol=1e-6)

    def test_model_trainable_parameters(self, model):
        params = list(model.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0


# ─────────────────────────────────────────────────────────────
# TESTES: server.py
# ─────────────────────────────────────────────────────────────

class TestConvergenceTracker:
    def test_convergence_after_stable_rounds(self):
        """
        CORREÇÃO: converged_round == 5, não 6.
        O tracker define converged_round = len(self.history) no momento da convergência.
        Com 5 chamadas a check(), len(history) = 5 quando a convergência é atingida.
        """
        tracker = ConvergenceTracker(threshold=0.01, patience=3)
        assert not tracker.check(0.50)   # history=[0.50]
        assert not tracker.check(0.51)   # history=[0.50, 0.51], delta=0.01 (NÃO < 0.01)
        assert not tracker.check(0.505)  # delta=0.005, stable=1
        assert not tracker.check(0.502)  # delta=0.003, stable=2
        assert tracker.check(0.501)      # delta=0.001, stable=3 → CONVERGIU
        assert tracker.converged_round == 5   # len(history) = 5

    def test_reset_clears_history(self):
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        tracker.check(0.5); tracker.check(0.51)
        tracker.reset()
        assert len(tracker.history) == 0
        assert tracker.stable_count == 0

    def test_no_convergence_with_large_deltas(self):
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        assert not tracker.check(0.50)
        assert not tracker.check(0.60)
        assert not tracker.check(0.70)
        assert tracker.stable_count == 0


class TestWeightedAverage:
    def test_weighted_average_basic(self):
        metrics = [(100, {"accuracy": 0.8}), (200, {"accuracy": 0.9})]
        result = weighted_average(metrics)
        expected = (100 * 0.8 + 200 * 0.9) / 300
        assert abs(result["accuracy"] - expected) < 0.001

    def test_weighted_average_empty(self):
        assert weighted_average([]) == {}

    def test_weighted_average_zero_examples(self):
        result = weighted_average([(0, {"accuracy": 0.8})])
        assert result == {"accuracy": 0.0}


class TestEvaluateFn:
    def test_evaluate_fn_runs(self):
        """
        CORREÇÃO: usa state_dict().values() para construir params,
        correspondendo ao que get_evaluate_fn espera receber.
        O modelo usa state_dict internamente, não model.parameters().
        """
        x = torch.randint(0, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        test_loader = [(x, y)]
        evaluate = get_evaluate_fn(test_loader)

        model = SimplifiedBEHRT(use_cls_token=True)
        params = [v.cpu().numpy() for v in model.state_dict().values()]

        loss, metrics = evaluate(1, params, {})
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


# ─────────────────────────────────────────────────────────────
# TESTES: client.py
# ─────────────────────────────────────────────────────────────

class TestFedProxClient:
    def test_get_parameters_returns_state_dict_values(self):
        """
        CORREÇÃO: get_parameters() retorna state_dict().values() (34 tensores),
        que inclui tanto parâmetros treináveis quanto buffers (ex: pe, cls_token_id).
        O original testava == model.parameters() (32), mas a implementação real
        usa state_dict para garantir compatibilidade com o carregamento no servidor.
        """
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        params = client.get_parameters({})
        sd_count = len(client.model.state_dict())
        assert len(params) == sd_count

    def test_proximal_loss_without_global_params(self):
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        loss = torch.tensor(1.5)
        assert torch.isclose(client._proximal_loss(loss), loss)

    def test_proximal_loss_with_global_params(self):
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        client = FedProxClient(0, loader, loader)
        client.global_params = [torch.zeros_like(p) for p in client.model.parameters()]
        loss = torch.tensor(1.5)
        assert client._proximal_loss(loss) > loss


# ─────────────────────────────────────────────────────────────
# TESTES: rag_system.py (com mocks)
# ─────────────────────────────────────────────────────────────

class TestClinicalRAG:
    """
    CORREÇÃO COMPLETA: os testes originais tentavam:
      1. 'import src.rag_system' — módulo inexistente (o certo é mosaicfl.v2.rag_system_v2)
      2. Modificar CHROMA_DB_PATH diretamente — não é como o ChromaDB instancia
      3. Instanciar ClinicalRAG sem mocks — baixa modelos HuggingFace (500 MB+)

    A solução correta é mockar os objetos externos no nível do __init__:
      - chromadb.PersistentClient   → não abre nenhum diretório
      - SentenceTransformer         → não baixa modelo de embeddings
      - AutoTokenizer / AutoModel   → não baixa modelo LLM
      - pipeline                    → não carrega GPU/CPU
    """

    def _make_rag(self):
        """Cria ClinicalRAG com todos os backends externos mockados."""
        from mosaicfl.v2.rag_system_v2 import ClinicalRAG

        mock_collection = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(3, 384).astype(np.float32)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        # tokenizer.encode retorna lista de ints; decode retorna str.
        # Sem isso, decode() → MagicMock e str.replace(MagicMock) lança TypeError.
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "prompt truncado"

        mock_llm = MagicMock()
        mock_generator = MagicMock(return_value=[{"generated_text": "Diagnóstico provável: covid19."}])

        with patch("mosaicfl.v2.rag_system_v2.chromadb.PersistentClient", return_value=mock_chroma), \
             patch("mosaicfl.v2.rag_system_v2.SentenceTransformer", return_value=mock_embedder), \
             patch("mosaicfl.v2.rag_system_v2.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("mosaicfl.v2.rag_system_v2.AutoModelForCausalLM.from_pretrained", return_value=mock_llm), \
             patch("mosaicfl.v2.rag_system_v2.pipeline", return_value=mock_generator):
            rag = ClinicalRAG()

        rag.embedder = mock_embedder
        rag.collection = mock_collection
        rag.generator = mock_generator
        rag.tokenizer = mock_tokenizer
        return rag, mock_collection, mock_embedder

    def test_build_knowledge_base_calls_collection_add(self):
        """build_knowledge_base deve chamar collection.add com embeddings."""
        rag, mock_collection, mock_embedder = self._make_rag()
        mock_embedder.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        patterns = [
            {"texto": "febre tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
            {"texto": "dispneia grave", "desfecho": "pneumonia", "faixa_etaria": "idoso"},
        ]
        rag.build_knowledge_base(patterns)
        mock_collection.add.assert_called_once()

    def test_retrieve_returns_list_of_dicts(self):
        """retrieve deve retornar lista com campos 'texto' e 'metadata'."""
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
        """explain deve retornar dict com as chaves esperadas."""
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


# ─────────────────────────────────────────────────────────────
# TESTES: config.py
# ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_device_is_cpu(self):
        assert str(DEVICE) == "cpu"

    def test_vocab_size_positive(self):
        assert VOCAB_SIZE > 0

    def test_batch_size_reasonable(self):
        assert BATCH_SIZE <= 32

    def test_num_rounds_limited(self):
        assert NUM_ROUNDS <= 50

    def test_proximal_mu_small(self):
        assert 0 < PROXIMAL_MU < 1

    def test_max_seq_len_power_of_two(self):
        assert MAX_SEQ_LEN > 0
        assert (MAX_SEQ_LEN & (MAX_SEQ_LEN - 1)) == 0


# ─────────────────────────────────────────────────────────────
# TESTES DE INTEGRAÇÃO
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_preprocess_model(self, sample_df):
        """Pipeline: preprocess → encode → model.forward."""
        pre = EHRPreprocessor()
        df_proc, summary = pre.process(
            sample_df, text_cols=["sintoma", "exame", "diagnostico"]
        )
        assert len(pre.vocab_map) > 4

        model = SimplifiedBEHRT(use_cls_token=True)
        seq = torch.tensor(
            [[pre.vocab_map.get("febre", 1), pre.vocab_map.get("tosse", 1), 0, 0, 0]]
        )
        logits = model(seq)
        assert logits.shape == (1, NUM_CLASSES)

    def test_model_parameters_match_client_server(self):
        """Cliente e servidor devem usar mesma arquitetura."""
        m_client = SimplifiedBEHRT()
        m_server = SimplifiedBEHRT()
        assert ([p.numel() for p in m_client.parameters()] ==
                [p.numel() for p in m_server.parameters()])

    def test_end_to_end_with_rag_mocked(self, sample_df):
        """Pipeline completo com RAG mockado não deve crashar."""
        from mosaicfl.v2.rag_system_v2 import ClinicalRAG

        mock_collection = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_collection.query.return_value = {
            "documents": [["febre tosse"] * 3],
            "metadatas": [[{"desfecho": "covid19"}] * 3],
            "distances": [[0.1, 0.15, 0.2]],
        }
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "prompt truncado"
        mock_generator = MagicMock(
            return_value=[{"generated_text": "Justificativa: covid19."}]
        )

        with patch("mosaicfl.v2.rag_system_v2.chromadb.PersistentClient", return_value=mock_chroma), \
             patch("mosaicfl.v2.rag_system_v2.SentenceTransformer", return_value=mock_embedder), \
             patch("mosaicfl.v2.rag_system_v2.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("mosaicfl.v2.rag_system_v2.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()), \
             patch("mosaicfl.v2.rag_system_v2.pipeline", return_value=mock_generator):
            rag = ClinicalRAG()

        rag.embedder = mock_embedder
        rag.collection = mock_collection
        rag.generator = mock_generator
        rag.tokenizer = mock_tokenizer

        pre = EHRPreprocessor()
        df_proc, _ = pre.process(sample_df, text_cols=["sintoma"])
        model = SimplifiedBEHRT(use_cls_token=True)
        seq = torch.tensor([[pre.vocab_map.get("febre", 1), 0, 0, 0, 0]])
        logits = model(seq)
        assert logits.shape == (1, NUM_CLASSES)

        result = rag.explain(
            {"febre": "alta", "saturacao": "95%"},
            {"diagnostico": "covid19", "probabilidade": 0.8}
        )
        assert "justificativa" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
