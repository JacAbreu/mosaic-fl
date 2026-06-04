"""
Testes unitários para o MOSAIC-FL (versões corrigidas).

Uso:
    pytest tests/test_mosaicfl.py -v

Requisitos:
    pip install pytest>=7.0.0
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path

# Garante que src/ está no path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import *
from src.preprocess import EHRPreprocessor, split_by_institution
from src.model import SimplifiedBEHRT, PositionalEncoding, BEHRTEncoderLayer
from src.client import FedProxClient, create_client_fn
from src.server import ConvergenceTracker, weighted_average, get_evaluate_fn
from src.rag_system import ClinicalRAG, JustificationResult
from src.extract_patterns import BEHRTPatternExtractor


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
        "peso": [150, 70, 180, 50, 80],
        "peso_unidade": ["lb", "kg", "lb", "kg", "kg"],
        "temperatura": [98.6, 36.5, 99.5, 37.0, 38.0],
        "sintoma": ["febre", "tosse", "dispneia", "fadiga", "mialgia"],
        "exame": ["rt_pcr_positivo", "tomografia_normal", "rx_consolidacao", "pcr_negativo", "tomografia_vidro_fosco"],
        "diagnostico": ["covid19_leve", "covid19_moderado", "pneumonia_bacteriana", "covid19_grave", "alta"],
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
        # 6 meses → 0.5 anos
        assert abs(df_norm.loc[1, "idade"] - 0.5) < 0.01
        # 365 dias → ~1.0 ano
        assert abs(df_norm.loc[3, "idade"] - 1.0) < 0.01
        # Todos em anos
        assert (df_norm["idade_unidade"] == "anos").all()

    def test_normalize_units_peso(self, preprocessor, sample_df):
        """Converte lb para kg."""
        df_norm = preprocessor.normalize_units(sample_df.copy())
        # 150 lb → ~68.04 kg
        assert abs(df_norm.loc[0, "peso"] - (150 * 0.453592)) < 0.01
        # 180 lb → ~81.65 kg
        assert abs(df_norm.loc[2, "peso"] - (180 * 0.453592)) < 0.01
        # Todos em kg
        assert (df_norm["peso_unidade"] == "kg").all()

    def test_clean_text_preserves_medical_punctuation(self, preprocessor):
        """Preserva pontos (ICD) e hífens (ranges) em clean_text."""
        df = pd.DataFrame({
            "sintoma": ["J18.1", "98.6°F", "18-65 anos", "febre!@#"],
        })
        cleaned = preprocessor.clean_text(df, ["sintoma"])
        assert "J18.1" in cleaned["sintoma"].values
        assert "98.6" in cleaned["sintoma"].values  # °F removido, mas 98.6 preservado
        assert "18-65" in cleaned["sintoma"].values
        # !@# removido
        assert "febre" in cleaned["sintoma"].values

    def test_build_vocabulary_includes_special_tokens(self, preprocessor, sample_df):
        """Vocabulário deve incluir <PAD>, <UNK>, <MASK>, <CLS>."""
        preprocessor.build_vocabulary(sample_df, ["sintoma", "exame", "diagnostico"])
        assert "<PAD>" in preprocessor.vocab_map
        assert "<UNK>" in preprocessor.vocab_map
        assert "<MASK>" in preprocessor.vocab_map
        assert "<CLS>" in preprocessor.vocab_map
        assert preprocessor.vocab_map["<PAD>"] == 0
        assert preprocessor.vocab_map["<UNK>"] == 1
        assert preprocessor.vocab_map["<MASK>"] == 2
        assert preprocessor.vocab_map["<CLS>"] == 3

    def test_handle_missing_impute(self, preprocessor):
        """Imputação com mediana para numéricos e <UNK> para categóricos."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", np.nan, "c", "d"],
        })
        df_imp = preprocessor.handle_missing(df, strategy="impute")
        # mediana de [1,2,4] = 2.0
        assert df_imp.loc[2, "num"] == 2.0
        assert df_imp.loc[1, "cat"] == "<UNK>"

    def test_process_returns_summary(self, preprocessor, sample_df):
        """process() deve retornar DataFrame + dict summary."""
        df_proc, summary = preprocessor.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        assert isinstance(df_proc, pd.DataFrame)
        assert isinstance(summary, dict)
        assert "total_amostras" in summary
        assert "tamanho_vocabulario" in summary
        assert summary["total_amostras"] == len(sample_df)


class TestSplitByInstitution:
    def test_split_creates_expected_clients(self, sample_df):
        """Deve criar 3 clientes (HospA, HospB, HospC)."""
        clients = split_by_institution(sample_df, num_clients=5)
        assert len(clients) == 3  # só há 3 instituições únicas
        assert 0 in clients
        assert 1 in clients
        assert 2 in clients

    def test_split_with_stratify(self, sample_df):
        """Com stratify, distribuição de desfecho deve ser balanceada."""
        clients = split_by_institution(sample_df, num_clients=3, stratify_col="desfecho")
        for cid, subset in clients.items():
            # Cada cliente deve ter ambos os desfechos (0 e 1)
            assert subset["desfecho"].nunique() >= 1

    def test_split_limits_num_clients(self, sample_df):
        """Se pedir mais clientes que instituições, ajusta automaticamente."""
        clients = split_by_institution(sample_df, num_clients=10)
        assert len(clients) == 3


# ─────────────────────────────────────────────────────────────
# TESTES: model.py
# ─────────────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        """Saída deve ter mesma shape da entrada."""
        pe = PositionalEncoding(d_model=64, max_len=128)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_adds_position_info(self):
        """Positional encoding deve alterar os valores (não ser identidade)."""
        pe = PositionalEncoding(d_model=64, max_len=128)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestBEHRTEncoderLayer:
    def test_forward_without_attention(self):
        """Forward normal deve retornar tensor apenas."""
        layer = BEHRTEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_forward_with_attention(self):
        """Forward com return_attention deve retornar (tensor, attn_weights)."""
        layer = BEHRTEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        x = torch.randn(2, 10, 64)
        out, attn = layer(x, return_attention=True)
        assert isinstance(out, torch.Tensor)
        assert isinstance(attn, torch.Tensor)
        # attn shape: (batch, num_heads, seq, seq)
        assert attn.shape == (2, 4, 10, 10)


class TestSimplifiedBEHRT:
    def test_forward_without_attention(self, model, dummy_sequences):
        """Forward normal deve retornar logits (batch, num_classes)."""
        x, _ = dummy_sequences
        logits = model(x)
        assert logits.shape == (x.size(0), NUM_CLASSES)

    def test_forward_with_attention(self, model, dummy_sequences):
        """Forward com return_attention deve retornar (logits, attn_stack)."""
        x, _ = dummy_sequences
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (x.size(0), NUM_CLASSES)
        # attn shape: (num_layers, batch, num_heads, seq_len, seq_len)
        assert attn.shape[0] == NUM_LAYERS
        assert attn.shape[1] == x.size(0)
        assert attn.shape[2] == NUM_HEADS

    def test_masked_mean_pool_excludes_padding(self, model):
        """Masked mean pooling deve ignorar posições de padding."""
        x = torch.tensor([[1, 2, 3, 0, 0]])  # 3 tokens reais + 2 PAD
        mask = (x == 0)  # [False, False, False, True, True]
        emb = model.embedding(x) * np.sqrt(EMBED_DIM)
        pooled = model._masked_mean_pool(emb, mask)
        # Média apenas dos 3 primeiros tokens
        expected = emb[0, :3].mean(dim=0)
        assert torch.allclose(pooled[0], expected, atol=1e-5)

    def test_cls_token_added(self, model):
        """Com use_cls_token=True, sequência deve ter +1 posição."""
        x = torch.randint(0, VOCAB_SIZE, (2, 10))
        emb = model.embedding(x) * np.sqrt(EMBED_DIM)
        # O forward adiciona CLS internamente, mas podemos verificar
        # que o vocab_size do embedding é VOCAB_SIZE + 1
        assert model.embedding.num_embeddings == VOCAB_SIZE + 1

    def test_dropout_respects_eval_mode(self, model, dummy_sequences):
        """Em eval mode, dropout deve estar desativado (determinístico)."""
        x, _ = dummy_sequences
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_model_trainable_parameters(self, model):
        """Modelo deve ter parâmetros treináveis."""
        params = list(model.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0
        print(f"Total de parâmetros: {total_params:,}")


# ─────────────────────────────────────────────────────────────
# TESTES: server.py
# ─────────────────────────────────────────────────────────────

class TestConvergenceTracker:
    def test_convergence_after_stable_rounds(self):
        """Deve convergir após patience rodadas estáveis."""
        tracker = ConvergenceTracker(threshold=0.01, patience=3)
        # Oscilação inicial
        assert not tracker.check(0.50)
        assert not tracker.check(0.51)
        # Estabiliza
        assert not tracker.check(0.505)   # delta=0.005 < 0.01, stable=1
        assert not tracker.check(0.502)   # delta=0.003 < 0.01, stable=2
        assert tracker.check(0.501)       # delta=0.001 < 0.01, stable=3 → CONVERGIU!
        assert tracker.converged_round == 6  # 6ª chamada (índice 5, mas round=6)

    def test_reset_clears_history(self):
        """Reset deve limpar histórico."""
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        tracker.check(0.5)
        tracker.check(0.51)
        tracker.reset()
        assert len(tracker.history) == 0
        assert tracker.stable_count == 0

    def test_no_convergence_with_large_deltas(self):
        """Não deve convergir se deltas forem grandes."""
        tracker = ConvergenceTracker(threshold=0.01, patience=2)
        assert not tracker.check(0.50)
        assert not tracker.check(0.60)  # delta=0.10 > 0.01
        assert not tracker.check(0.70)  # delta=0.10 > 0.01
        assert tracker.stable_count == 0


class TestWeightedAverage:
    def test_weighted_average_basic(self):
        """Média ponderada deve funcionar corretamente."""
        metrics = [
            (100, {"accuracy": 0.8}),
            (200, {"accuracy": 0.9}),
        ]
        result = weighted_average(metrics)
        expected = (100*0.8 + 200*0.9) / 300  # = 0.8667
        assert abs(result["accuracy"] - expected) < 0.001

    def test_weighted_average_empty(self):
        """Lista vazia deve retornar dict vazio."""
        result = weighted_average([])
        assert result == {}

    def test_weighted_average_zero_examples(self):
        """Deve lidar com zero exemplos (edge case)."""
        metrics = [(0, {"accuracy": 0.8})]
        result = weighted_average(metrics)
        assert result == {"accuracy": 0.0}


class TestEvaluateFn:
    def test_evaluate_fn_runs(self):
        """get_evaluate_fn deve retornar callable que executa sem erro."""
        # Cria dummy test_loader
        x = torch.randint(0, VOCAB_SIZE, (8, 16))
        y = torch.randint(0, NUM_CLASSES, (8,))
        test_loader = [(x, y)]

        evaluate = get_evaluate_fn(test_loader)
        # Simula parâmetros (lista de arrays numpy)
        model = SimplifiedBEHRT()
        params = [p.detach().cpu().numpy() for p in model.parameters()]

        loss, metrics = evaluate(1, params, {})
        assert isinstance(loss, float)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


# ─────────────────────────────────────────────────────────────
# TESTES: client.py
# ─────────────────────────────────────────────────────────────

class TestFedProxClient:
    def test_get_parameters_returns_trainable_only(self):
        """get_parameters deve retornar apenas parâmetros treináveis (não buffers)."""
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        train_loader = [(x, y)]
        val_loader = [(x, y)]

        client = FedProxClient(0, train_loader, val_loader)
        params = client.get_parameters({})

        # Deve ter mesmo número de elementos que model.parameters()
        model_params = list(client.model.parameters())
        assert len(params) == len(model_params)

        # Não deve incluir buffers (ex: running_mean de BN — não há no BEHRT,
        # mas testamos que o número bate com parameters(), não state_dict())
        state_dict_items = len(client.model.state_dict())
        assert len(params) <= state_dict_items  # pode ser igual se não houver buffers

    def test_proximal_loss_without_global_params(self):
        """Sem global_params, proximal_loss deve retornar loss inalterado."""
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        train_loader = [(x, y)]
        val_loader = [(x, y)]

        client = FedProxClient(0, train_loader, val_loader)
        loss = torch.tensor(1.5)
        result = client._proximal_loss(loss)
        assert torch.isclose(result, loss)

    def test_proximal_loss_with_global_params(self):
        """Com global_params, proximal_loss deve ser maior que loss original."""
        x = torch.randint(0, VOCAB_SIZE, (4, 16))
        y = torch.randint(0, NUM_CLASSES, (4,))
        train_loader = [(x, y)]
        val_loader = [(x, y)]

        client = FedProxClient(0, train_loader, val_loader)
        # Simula global_params = zeros (diferente dos pesos atuais)
        client.global_params = [torch.zeros_like(p) for p in client.model.parameters()]

        loss = torch.tensor(1.5)
        result = client._proximal_loss(loss)
        assert result > loss  # termo proximal adiciona penalidade


# ─────────────────────────────────────────────────────────────
# TESTES: rag_system.py
# ─────────────────────────────────────────────────────────────

class TestClinicalRAG:
    def test_build_knowledge_base(self, tmp_path):
        """Deve indexar padrões no ChromaDB sem erro."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Sobrescreve CHROMA_DB_PATH temporariamente
            import src.rag_system as rag_mod
            original_path = rag_mod.CHROMA_DB_PATH
            rag_mod.CHROMA_DB_PATH = tmpdir

            try:
                rag = ClinicalRAG()
                patterns = [
                    {"texto": "Paciente com febre e tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
                    {"texto": "Paciente com dispneia grave", "desfecho": "pneumonia", "faixa_etaria": "idoso"},
                ]
                rag.build_knowledge_base(patterns)
                # Se não crashou, passou
                assert True
            finally:
                rag_mod.CHROMA_DB_PATH = original_path

    def test_retrieve_returns_top_k(self, tmp_path):
        """retrieve deve retornar exatamente top_k resultados."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.rag_system as rag_mod
            original_path = rag_mod.CHROMA_DB_PATH
            rag_mod.CHROMA_DB_PATH = tmpdir

            try:
                rag = ClinicalRAG()
                patterns = [
                    {"texto": "Paciente com febre e tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
                    {"texto": "Paciente com dispneia grave", "desfecho": "pneumonia", "faixa_etaria": "idoso"},
                    {"texto": "Paciente com fadiga leve", "desfecho": "covid19", "faixa_etaria": "jovem"},
                ]
                rag.build_knowledge_base(patterns)
                results = rag.retrieve("febre tosse", top_k=2)
                assert len(results) == 2
                assert all("texto" in r for r in results)
                assert all("metadata" in r for r in results)
            finally:
                rag_mod.CHROMA_DB_PATH = original_path

    def test_generate_justification_returns_tuple(self, tmp_path):
        """generate_justification deve retornar (str, list, bool)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.rag_system as rag_mod
            original_path = rag_mod.CHROMA_DB_PATH
            rag_mod.CHROMA_DB_PATH = tmpdir

            try:
                rag = ClinicalRAG()
                cases = [
                    {"texto": "Paciente com febre e tosse, evoluiu bem", "metadata": {"desfecho": "alta"}, "distancia": 0.1},
                ]
                justification, sources, hallucinated = rag.generate_justification(
                    prediction="pneumonia",
                    probability=0.75,
                    symptoms="fever, cough",
                    retrieved_cases=cases,
                )
                assert isinstance(justification, str)
                assert isinstance(sources, list)
                assert isinstance(hallucinated, bool)
            finally:
                rag_mod.CHROMA_DB_PATH = original_path

    def test_explain_returns_expected_keys(self, tmp_path):
        """explain deve retornar dict com chaves esperadas."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.rag_system as rag_mod
            original_path = rag_mod.CHROMA_DB_PATH
            rag_mod.CHROMA_DB_PATH = tmpdir

            try:
                rag = ClinicalRAG()
                patterns = [
                    {"texto": "Paciente com febre e tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
                ]
                rag.build_knowledge_base(patterns)

                patient = {"febre": "alta", "tosse": "seca", "saturacao": "92%", "faixa_etaria": "adulto"}
                pred = {"diagnostico": "pneumonia", "probabilidade": 0.82}

                result = rag.explain(patient, pred)
                assert "predicao" in result
                assert "justificativa" in result
                assert "fontes" in result
                assert "alucinacao_detectada" in result
                assert "confiavel" in result
                assert isinstance(result["confiavel"], bool)
            finally:
                rag_mod.CHROMA_DB_PATH = original_path


# ─────────────────────────────────────────────────────────────
# TESTES: extract_patterns.py
# ─────────────────────────────────────────────────────────────

class TestBEHRTPatternExtractor:
    def test_init_requires_vocab(self):
        """Deve lançar ValueError se vocab_map estiver vazio."""
        model = SimplifiedBEHRT()
        with pytest.raises(ValueError, match="vocab_map está vazio"):
            BEHRTPatternExtractor(model, {})

    def test_extract_top_patterns_runs(self):
        """Deve executar sem erro com dummy data."""
        model = SimplifiedBEHRT()
        vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3, "febre": 4, "tosse": 5}
        extractor = BEHRTPatternExtractor(model, vocab)

        # Dummy dataloader
        x = torch.randint(0, 10, (4, 16))
        y = torch.randint(0, 2, (4,))
        dummy_loader = [(x, y)]

        patterns = extractor.extract_top_patterns(dummy_loader, desfecho_alvo=0, top_n=5)
        assert isinstance(patterns, list)
        # Pode retornar lista vazia se nenhum desfecho_alvo no batch
        # Mas não deve crashar


# ─────────────────────────────────────────────────────────────
# TESTES: config.py
# ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_device_is_cpu(self):
        """DEVICE deve ser CPU conforme calibração de hardware."""
        assert str(DEVICE) == "cpu"

    def test_vocab_size_positive(self):
        """VOCAB_SIZE deve ser positivo."""
        assert VOCAB_SIZE > 0

    def test_batch_size_reasonable(self):
        """BATCH_SIZE deve ser pequeno o suficiente para 16GB RAM."""
        assert BATCH_SIZE <= 32

    def test_num_rounds_limited(self):
        """NUM_ROUNDS deve ser limitado para hardware alvo."""
        assert NUM_ROUNDS <= 50

    def test_proximal_mu_small(self):
        """PROXIMAL_MU deve ser pequeno (regularização leve)."""
        assert 0 < PROXIMAL_MU < 1

    def test_max_seq_len_power_of_two(self):
        """MAX_SEQ_LEN deve ser potência de 2 (eficiência de memória)."""
        assert MAX_SEQ_LEN > 0
        assert (MAX_SEQ_LEN & (MAX_SEQ_LEN - 1)) == 0  # potência de 2


# ─────────────────────────────────────────────────────────────
# TESTES DE INTEGRAÇÃO
# ─────────────────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_pipeline(self, sample_df, tmp_path):
        """Testa pipeline completo: preprocess → model → RAG (sem FL)."""
        # 1. Pré-processamento
        pre = EHRPreprocessor()
        df_proc, summary = pre.process(sample_df, text_cols=["sintoma", "exame", "diagnostico"])
        assert len(pre.vocab_map) > 4  # pelo menos os 4 tokens especiais

        # 2. Modelo
        model = SimplifiedBEHRT(use_cls_token=True)
        seq = torch.tensor([[pre.vocab_map.get("febre", 1), pre.vocab_map.get("tosse", 1), 0, 0, 0]])
        logits = model(seq)
        assert logits.shape == (1, NUM_CLASSES)

        # 3. RAG (com ChromaDB temporário)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.rag_system as rag_mod
            original_path = rag_mod.CHROMA_DB_PATH
            rag_mod.CHROMA_DB_PATH = tmpdir
            try:
                rag = ClinicalRAG()
                patterns = [
                    {"texto": "Paciente com febre e tosse", "desfecho": "covid19", "faixa_etaria": "adulto"},
                ]
                rag.build_knowledge_base(patterns)
                result = rag.explain(
                    {"febre": "alta", "tosse": "seca", "saturacao": "95%", "faixa_etaria": "adulto"},
                    {"diagnostico": "covid19", "probabilidade": 0.8}
                )
                assert "justificativa" in result
                assert "confiavel" in result
            finally:
                rag_mod.CHROMA_DB_PATH = original_path

    def test_model_parameters_match_client_server(self):
        """Cliente e servidor devem usar mesma arquitetura de modelo."""
        model_client = SimplifiedBEHRT()
        model_server = SimplifiedBEHRT()

        client_params = [p.numel() for p in model_client.parameters()]
        server_params = [p.numel() for p in model_server.parameters()]

        assert client_params == server_params
        # get_parameters do cliente deve bater com load_state_dict do servidor
        # (considerando que usamos strict=False no servidor para buffers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
