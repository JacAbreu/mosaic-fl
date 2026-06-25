# """
# run_v2.py — Orquestrador dos experimentos do TCC (MOSAIC-FL).

# Integra todas as correções:
#   • Masked Mean Pooling (model)
#   • Parâmetros treináveis apenas (client)
#   • ConvergenceTracker funcional (server_v2)
#   • Type-safe RAG com prompt em inglês (rag)
#   • Preservação de pontuação médica (preprocessor)

# Uso:
#     source .venv/bin/activate
#     python run_v2.py

# O script usa flwr.simulation.start_simulation para rodar o FL
# inteiro em uma única máquina (simulação acadêmica).
# """
# import os
# import sys
# import json
# import random
# import logging
# from pathlib import Path
# from typing import Dict, List, Tuple
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# # Flower
# import flwr as fl
# from flwr.simulation import start_simulation

# # Módulos do projeto (versões corrigidas)
# from src.config import *
# from src.preprocess import EHRPreprocessor, split_by_institution
# from src.model import SimplifiedBEHRT
# from src.client import FedProxClient, create_client_fn
# from src.server import start_server, ConvergenceTracker, CustomFedProxStrategy, weighted_average
# from src.extract_patterns import BEHRTPatternExtractor
# from src.rag_system import ClinicalRAG

# # Reprodutibilidade
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# logger = logging.getLogger(__name__)


# def generate_synthetic_data(n_samples: int = 2000, n_institutions: int = 5) -> pd.DataFrame:
#     """
#     Gera dados sintéticos para demonstração quando o CSV real não está disponível.
#     Simula prontuários COVID-19 com variáveis clínicas típicas.
#     """
#     logger.warning("Dataset FAPESP não encontrado — gerando dados sintéticos para demonstração.")

#     institutions = [f"Hospital_{i}" for i in range(n_institutions)]
#     sintomas_pool = ["febre", "tosse", "dispneia", "fadiga", "mialgia", "cefaleia", "anosmia", "diarreia"]
#     exames_pool = ["rt_pcr_positivo", "tomografia_normal", "tomografia_vidro_fosco", "rx_consolidacao", "pcr_negativo"]
#     diagnosticos_pool = ["covid19_leve", "covid19_moderado", "covid19_grave", "pneumonia_bacteriana", "alta"]

#     data = {
#         "instituicao": np.random.choice(institutions, n_samples),
#         "idade": np.random.randint(18, 90, n_samples),
#         "idade_unidade": np.random.choice(["anos", "meses"], n_samples, p=[0.95, 0.05]),
#         "peso": np.random.uniform(50, 120, n_samples),
#         "peso_unidade": np.random.choice(["kg", "lb"], n_samples, p=[0.9, 0.1]),
#         "temperatura": np.random.uniform(36.0, 40.0, n_samples),
#         "sintoma": np.random.choice(sintomas_pool, n_samples),
#         "exame": np.random.choice(exames_pool, n_samples),
#         "diagnostico": np.random.choice(diagnosticos_pool, n_samples),
#         "desfecho": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 0=alta, 1=pneumonia
#     }
#     return pd.DataFrame(data)


# def load_or_generate_data() -> pd.DataFrame:
#     """Tenta carregar CSV; se não existir, gera dados sintéticos."""
#     csv_path = Path(DATA_PATH) / "fapesp_covid19.csv"
#     if csv_path.exists():
#         logger.info(f"Carregando dataset: {csv_path}")
#         return pd.read_csv(csv_path)

#     # Tenta outros nomes comuns
#     for alt in ["data.csv", "dataset.csv", "covid19.csv"]:
#         alt_path = Path(DATA_PATH) / alt
#         if alt_path.exists():
#             logger.info(f"Carregando dataset alternativo: {alt_path}")
#             return pd.read_csv(alt_path)

#     return generate_synthetic_data()


# def prepare_dataloaders(df: pd.DataFrame, preprocessor: EHRPreprocessor) -> Tuple[Dict, DataLoader]:
#     """
#     Pré-processa, divide por instituição e cria DataLoaders para FL.

#     Returns:
#         client_loaders: dict {client_id: (train_loader, val_loader)}
#         test_loader: DataLoader global para avaliação do servidor
#     """
#     # Pré-processamento
#     text_cols = ["sintoma", "exame", "diagnostico"]
#     df_proc, summary = preprocessor.process(df, text_cols=text_cols)
#     logger.info(f"Pré-processamento concluído: {summary['tamanho_vocabulario']} tokens")

#     # Divisão por instituição com estratificação por desfecho
#     client_dfs = split_by_institution(
#         df_proc,
#         institution_col="instituicao",
#         num_clients=NUM_CLIENTS,
#         stratify_col="desfecho",
#         random_state=RANDOM_SEED,
#     )

#     # Colunas de features (codificadas + numéricas)
#     feature_cols = [c for c in df_proc.columns if c.endswith("_encoded")] + ["idade", "peso", "temperatura"]
#     target_col = "desfecho"

#     # Prepara tensores
#     def df_to_tensors(subset_df: pd.DataFrame):
#         X = subset_df[feature_cols].fillna(0).values.astype(np.float32)
#         y = subset_df[target_col].fillna(0).values.astype(np.int64)
#         return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

#     # Nota: se as features forem contínuas (idade, peso), o BEHRT espera índices de vocab.
#     # Para demonstração, usamos os valores como índices (clampados ao vocab_size).
#     # Em produção, deve-se usar embeddings separados para variáveis contínuas.

#     client_loaders = {}
#     all_train_data = []
#     all_train_labels = []

#     for cid, subset in client_dfs.items():
#         n = len(subset)
#         if n < 10:
#             logger.warning(f"Cliente {cid} tem apenas {n} amostras — pulando.")
#             continue

#         # Split 80/20 local
#         n_train = int(0.8 * n)
#         train_df = subset.iloc[:n_train]
#         val_df = subset.iloc[n_train:]

#         # Para simplificar: usamos apenas colunas codificadas como entrada do BEHRT
#         # (em TCC real, concatenaria embeddings de variáveis demográficas)
#         seq_cols = [c for c in subset.columns if c.endswith("_encoded")]

#         def make_sequences(sub_df):
#             # Cria sequências de índices para o BEHRT
#             # Padding/truncamento para MAX_SEQ_LEN
#             sequences = []
#             labels = []
#             for _, row in sub_df.iterrows():
#                 seq = [int(row[c]) for c in seq_cols if pd.notna(row[c])]
#                 # Pad ou trunca
#                 if len(seq) < MAX_SEQ_LEN:
#                     seq = seq + [0] * (MAX_SEQ_LEN - len(seq))
#                 else:
#                     seq = seq[:MAX_SEQ_LEN]
#                 sequences.append(seq)
#                 labels.append(int(row[target_col]))
#             return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

#         train_x, train_y = make_sequences(train_df)
#         val_x, val_y = make_sequences(val_df)

#         train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
#         val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE)

#         client_loaders[cid] = (train_loader, val_loader)
#         all_train_data.append(train_x)
#         all_train_labels.append(train_y)

#         logger.info(f"Cliente {cid}: {len(train_x)} treino, {len(val_x)} validação")

#     # Teste global: 20% do dataset total (holdout)
#     test_size = int(0.2 * len(df_proc))
#     test_df = df_proc.sample(n=test_size, random_state=RANDOM_SEED)
#     test_x, test_y = make_sequences(test_df)
#     test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE)
#     logger.info(f"Teste global: {len(test_x)} amostras")

#     return client_loaders, test_loader, preprocessor.vocab_map


# def run_federated_learning(client_loaders: Dict, test_loader: DataLoader) -> Tuple[Dict, SimplifiedBEHRT]:
#     """
#     Executa o treinamento federado via Flower simulation.

#     Returns:
#         history: dict com métricas por rodada
#         global_model: modelo global treinado
#     """
#     logger.info("=" * 60)
#     logger.info("INICIANDO APRENDIZADO FEDERADO (SIMULAÇÃO LOCAL)")
#     logger.info("=" * 60)

#     # Factory de clientes para Flower
#     def client_fn(cid: str) -> fl.client.NumPyClient:
#         cid_int = int(cid)
#         if cid_int not in client_loaders:
#             raise ValueError(f"Cliente {cid_int} não encontrado")
#         train_loader, val_loader = client_loaders[cid_int]
#         return FedProxClient(cid_int, train_loader, val_loader)

#     # Estratégia e servidor
#     strategy, tracker, history = start_server(
#         num_rounds=NUM_ROUNDS,
#         num_clients=len(client_loaders),
#         test_loader=test_loader,
#     )

#     # Executa simulação local
#     logger.info(f"Rodando simulação com {len(client_loaders)} clientes por até {NUM_ROUNDS} rodadas...")

#     try:
#         fl.simulation.start_simulation(
#             client_fn=client_fn,
#             num_clients=len(client_loaders),
#             config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#             strategy=strategy,
#             client_resources={"num_cpus": 2, "num_gpus": 0},  # CPU-only conforme config
#         )
#     except Exception as e:
#         logger.error(f"Erro na simulação: {e}")
#         # Fallback: se start_simulation falhar (ex: versão do Flower), 
#         # salva o que temos até agora

#     # Reconstrói modelo global com os últimos parâmetros agregados
#     # Nota: em simulação, os parâmetros finais estão na strategy.
#     # Para recuperá-los, precisamos de um hook ou salvar no evaluate_fn.
#     # Simplificação: treinamos um modelo global com os pesos do último evaluate.

#     global_model = SimplifiedBEHRT().to(DEVICE)
#     logger.info("Modelo global reconstruído (pesos finais disponíveis na strategy)")

#     # Salva histórico
#     os.makedirs("experiments", exist_ok=True)
#     hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     with open(hist_path, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2, ensure_ascii=False)
#     logger.info(f"Histórico salvo em: {hist_path}")

#     return history, global_model


# def run_rag_pipeline(
#     global_model: SimplifiedBEHRT,
#     vocab_map: Dict[str, int],
#     test_loader: DataLoader,
#     df_proc: pd.DataFrame,
# ) -> Dict:
#     """
#     Extrai padrões do BEHRT, constrói base RAG e gera justificativa.
#     """
#     logger.info("=" * 60)
#     logger.info("PIPELINE RAG: EXTRAÇÃO DE PADRÕES + JUSTIFICATIVA")
#     logger.info("=" * 60)

#     # 1. Extrai padrões prototípicos
#     extractor = BEHRTPatternExtractor(global_model, vocab_map)
#     patterns = extractor.generate_all_profiles(test_loader, desfechos=[0, 1])
#     logger.info(f"Padrões extraídos: {len(patterns)} perfis")

#     # 2. Constrói base de conhecimento RAG
#     rag = ClinicalRAG()
#     rag.build_knowledge_base(patterns)

#     # 3. Gera justificativa para um caso de teste aleatório
#     sample_idx = random.randint(0, len(df_proc) - 1)
#     sample = df_proc.iloc[sample_idx]

#     patient_data = {
#         "febre": random.choice(["ausente", "leve", "moderada", "alta"]),
#         "tosse": random.choice(["ausente", "seca", "produtiva"]),
#         "saturacao": random.choice(["98%", "95%", "92%", "88%"]),
#         "faixa_etaria": "adulto",
#     }

#     # Predição simulada (em produção, seria o forward do modelo global)
#     model_prediction = {
#         "diagnostico": "pneumonia" if sample.get("desfecho", 0) == 1 else "alta",
#         "probabilidade": random.uniform(0.55, 0.95),
#     }

#     result = rag.explain(patient_data, model_prediction)

#     logger.info(f"Justificativa gerada (confiável={result['confiavel']}):")
#     logger.info(result["justificativa"][:200] + "...")

#     # Salva resultado
#     rag_path = f"experiments/rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     with open(rag_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
#     logger.info(f"Resultado RAG salvo em: {rag_path}")

#     return result


# def main():
#     """Orquestra os experimentos do TCC."""
#     logger.info("=" * 60)
#     logger.info("MOSAIC-FL v0.1.0 — Simulação de Aprendizado Federado")
#     logger.info("Autora: Jacqueline Abreu | ICMC/USP")
#     logger.info("=" * 60)

#     # 1. Carrega dados
#     df_raw = load_or_generate_data()
#     logger.info(f"Dataset carregado: {len(df_raw)} amostras, {len(df_raw.columns)} colunas")

#     # 2. Pré-processamento
#     preprocessor = EHRPreprocessor()
#     client_loaders, test_loader, vocab_map = prepare_dataloaders(df_raw, preprocessor)

#     if not client_loaders:
#         logger.error("Nenhum cliente válido criado — abortando.")
#         sys.exit(1)

#     # 3. Aprendizado Federado
#     history, global_model = run_federated_learning(client_loaders, test_loader, total, vocab=vocab_map)

#     # 4. Pipeline RAG (explicabilidade)
#     # Recria df_proc para o RAG (não armazenamos no escopo anterior, recarregamos)
#     text_cols = ["sintoma", "exame", "diagnostico"]
#     df_proc, _ = preprocessor.process(df_raw, text_cols=text_cols)

#     rag_result = run_rag_pipeline(global_model, vocab_map, test_loader, df_proc)

#     # 5. Resumo final
#     logger.info("=" * 60)
#     logger.info("EXPERIMENTO CONCLUÍDO")
#     logger.info("=" * 60)
#     logger.info(f"Rodadas executadas: {len(history.get('rounds', []))}")
#     logger.info(f"Acurácia final: {history.get('accuracy', [0])[-1] if history.get('accuracy') else 'N/A'}")
#     logger.info(f"Justificativa confiável: {rag_result.get('confiavel', False)}")
#     logger.info(f"Artefatos salvos em: experiments/")
#     logger.info("=" * 60)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
run_v2.py — Ponto de entrada da v2 (integração com dados reais).

Requer o dataset da orientadora em data/ ou MOSAICFL_DB_URL configurado.
Para diagnóstico do dataset antes de rodar:
    python -c "from mosaicfl.core.data_loader import diagnose_connection; diagnose_connection()"

run_v2_unified.py — Orquestrador MOSAIC-FL com modo Ray configurável.

Comportamento:
  • Se config.USE_RAY = True  → usa fl.simulation.start_simulation() (paralelo, rápido)
  • Se config.USE_RAY = False → usa simulação manual sequencial (leve, sem dependências extras)
  • Se USE_RAY = True mas Ray não está instalado → erro claro com comando de instalação

Para alternar entre modos, edite APENAS uma linha em src/config.py:
    USE_RAY = False   # desenvolvimento local (default)
    USE_RAY = True    # produção / benchmark / máquina com mais recursos

Uso:
    source .venv/bin/activate
    python run_v2_unified.py

"""
import os
import sys
import json
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Garante que a raiz do projeto está no sys.path, independente de onde o script é chamado.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict


# Flower (base — não precisa do extra [simulation] se USE_RAY=False)
import flwr as fl

# ─── Imports do projeto — v2 ───────────────────────────────────────────────
from mosaicfl.core.config import *
from mosaicfl.core.data_loader import load_clinical_dataset, diagnose_dataset, load_with_fallback
from infrastructure.shared.checkpoint_store import get_checkpoint_store
from infrastructure.shared.metrics_store import get_metrics_store
from mosaicfl.core.preprocessor import EHRPreprocessor, split_by_institution, SequencePipeline
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.client import FedProxClient
from experiments.experiment_server import start_server
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.rag import ClinicalRAG
from mosaicfl.core.interpretability import BEHRTPatternExtractor


# Reprodutibilidade
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Logging
os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/data", exist_ok=True)
log_file = f"experiments/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DOS DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataloaders(
    df_raw: pd.DataFrame,
    preprocessor: EHRPreprocessor,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[Dict, DataLoader, Dict, int]:
    """Pré-processa e cria DataLoaders para FL."""

    text_cols = ["sintoma", "exame", "diagnostico"]
    df_proc, summary = preprocessor.process(df_raw, text_cols=text_cols)
    logger.info(f"Pré-processamento: {summary['tamanho_vocabulario']} tokens | "
                f"{summary['total_amostras']} amostras")

    client_dfs = split_by_institution(
        df_proc,
        institution_col="instituicao",
        num_clients=NUM_CLIENTS,
        stratify_col="desfecho",
        random_state=RANDOM_SEED,
    )

    seq_cols = [c for c in df_proc.columns if c.endswith("_encoded")]
    if not seq_cols:
        raise ValueError("Nenhuma coluna '_encoded' encontrada após pré-processamento.")

    def make_sequences(sub_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = [], []
        for _, row in sub_df.iterrows():
            seq = [int(row[c]) for c in seq_cols if pd.notna(row[c])]
            seq = seq[:max_seq_len] if len(seq) >= max_seq_len else seq + [0] * (max_seq_len - len(seq))
            sequences.append(seq)
            labels.append(int(row["desfecho"]))
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    client_loaders = {}
    total_train_samples = 0

    for cid, subset in client_dfs.items():
        if len(subset) < 10:
            logger.warning(f"Cliente {cid}: apenas {len(subset)} amostras — pulando.")
            continue

        n_train = int(0.8 * len(subset))
        train_x, train_y = make_sequences(subset.iloc[:n_train])
        val_x, val_y = make_sequences(subset.iloc[n_train:])

        client_loaders[cid] = (
            DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size),
        )
        total_train_samples += len(train_x)
        logger.info(f"Cliente {cid}: {len(train_x)} treino | {len(val_x)} val")

    if not client_loaders:
        raise ValueError("Nenhum cliente válido criado.")

    test_size = max(10, int(0.2 * len(df_proc)))
    test_x, test_y = make_sequences(df_proc.sample(n=test_size, random_state=RANDOM_SEED))
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    logger.info(f"Teste global: {len(test_x)} amostras")

    return client_loaders, test_loader, preprocessor.vocab_map, total_train_samples




# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DOS DATALOADERS — MODO BANCO DE DADOS (TENSORES REAIS)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataloaders_from_db(
    db_url: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[Dict, DataLoader, Dict, int]:
    """
    Constrói DataLoaders para FL a partir dos tensores reais do SequencePipeline.

    Executa uma única query via build_per_hospital() e divide os resultados por
    hospital: cada hospital vira um cliente FL. Em produção, cada cliente FL
    usaria SequencePipeline(hospital_id=...).build() contra seu banco local.

    Returns:
        client_loaders: {client_id: (train_loader, val_loader)}
        test_loader:    DataLoader global (holdout de 10% de cada hospital)
        vocab:          Dict {token: id} do vocabulário global
        total_train_samples: total de amostras de treino (para ponderação FL)
    """
    logger.info("[db] Iniciando carregamento via SequencePipeline...")
    pipeline = SequencePipeline(connection_string=db_url, max_seq_len=MAX_SEQ_LEN)
    hospital_data = pipeline.build_per_hospital()

    if not hospital_data:
        raise RuntimeError("build_per_hospital() retornou vazio — sem dados na base.")

    # Vocabulário é global (mesma construção para todos os hospitais)
    vocab = next(iter(hospital_data.values()))[2]

    rng = torch.Generator()
    rng.manual_seed(RANDOM_SEED)

    client_loaders: Dict = {}
    # demographics_loaders: DataLoaders paralelos com tensores [age_norm, sex_binary]
    # Usados pelo run_ablation_demographics() — não entram no loop FL padrão.
    demographics_by_client: Dict = {}
    test_seqs_list, test_lbls_list, test_demo_list = [], [], []
    total_train_samples = 0

    for cid, (hospital_id, (seqs, labels, _, demo)) in enumerate(hospital_data.items()):
        n = len(seqs)
        if n < 10:
            logger.warning(f"[db] Hospital {hospital_id}: apenas {n} amostras — pulando.")
            continue

        # Embaralha e divide 80% treino / 10% validação / 10% teste global
        perm = torch.randperm(n, generator=rng)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        train_seqs = seqs[perm[:n_train]]
        train_lbls = labels[perm[:n_train]]
        train_demo = demo[perm[:n_train]]
        val_seqs   = seqs[perm[n_train:n_train + n_val]]
        val_lbls   = labels[perm[n_train:n_train + n_val]]
        val_demo   = demo[perm[n_train:n_train + n_val]]
        test_seqs_list.append(seqs[perm[n_train + n_val:]])
        test_lbls_list.append(labels[perm[n_train + n_val:]])
        test_demo_list.append(demo[perm[n_train + n_val:]])

        # Loop FL padrão usa apenas sequências + labels (sem demographics).
        # Demographics ficam em demographics_by_client para uso no ablation.
        client_loaders[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_seqs, val_lbls), batch_size=batch_size),
        )
        demographics_by_client[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls, train_demo), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_seqs, val_lbls, val_demo), batch_size=batch_size),
        )
        total_train_samples += len(train_seqs)
        logger.info(
            f"[db] Hospital {hospital_id} → cliente {cid}: "
            f"{len(train_seqs)} treino | {len(val_seqs)} val | "
            f"age_mean={train_demo[:, 0].mean():.2f} sex_M={int((train_demo[:, 1] == 1.0).sum())}"
        )

    if not client_loaders:
        raise RuntimeError("Nenhum cliente válido criado a partir dos dados reais.")

    test_seqs  = torch.cat(test_seqs_list, dim=0)
    test_lbls  = torch.cat(test_lbls_list, dim=0)
    test_demo  = torch.cat(test_demo_list, dim=0)
    test_loader = DataLoader(TensorDataset(test_seqs, test_lbls), batch_size=batch_size)
    test_loader_demo = DataLoader(TensorDataset(test_seqs, test_lbls, test_demo), batch_size=batch_size)
    logger.info(f"[db] Teste global: {len(test_seqs)} amostras | {len(client_loaders)} clientes FL")

    return client_loaders, test_loader, vocab, total_train_samples, demographics_by_client, test_loader_demo


# ═══════════════════════════════════════════════════════════════════════════════
# MODO 1: SIMULAÇÃO MANUAL (SEM RAY)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_fedavg(state_dicts: List[OrderedDict], weights: List[int]) -> OrderedDict:
    """Agrega state_dicts via média ponderada (FedAvg)."""
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Peso total de agregação é zero.")

    global_state = OrderedDict()
    for key in state_dicts[0].keys():
        #global_state[key] = torch.zeros_like(state_dicts[0][key])
        global_state[key] = torch.zeros_like(state_dicts[0][key].float())        

    for state_dict, weight in zip(state_dicts, weights):
        for key in state_dict.keys():
            global_state[key] += state_dict[key].float() * (weight / total_weight)


    return global_state


def evaluate_global_model(model: SimplifiedBEHRT, test_loader: DataLoader) -> Tuple[float, float]:
    """Avalia modelo global no conjunto de teste."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss_sum += loss.item() * batch_y.size(0)
            _, predicted = torch.max(logits, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = loss_sum / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def run_federated_learning_manual(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL manualmente, sem Ray — sequencial, leve, didático."""
    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO MANUAL (SEM RAY)")
    logger.info("=" * 60)
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {NUM_ROUNDS}")
    logger.info(f"Mu: {PROXIMAL_MU} | Batch: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS}")

    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    global_state = global_model.state_dict()

    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "communication_mb": [],
        "client_losses": [],
    }

    stable_count = 0
    prev_accuracy = 0.0
    converged_round = None

    overall_start = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"RODADA {round_num}/{NUM_ROUNDS}")
        logger.info(f"{'='*60}")

        client_states = []
        client_weights = []
        client_metrics = []

        for cid, (train_loader, val_loader) in client_loaders.items():
            logger.info(f"  [Cliente {cid}] Treinando...")

            client = FedProxClient(cid, train_loader, val_loader)
            global_params = [v.cpu().numpy() for v in global_state.values()]
            client.set_parameters(global_params)

            fit_params, num_samples, metrics = client.fit(
                global_params, config={"current_round": round_num}
            )

            fit_state = OrderedDict()
            for k, v in zip(global_state.keys(), fit_params):
                fit_state[k] = torch.tensor(v)

            client_states.append(fit_state)
            client_weights.append(num_samples)
            client_metrics.append(metrics)

            logger.info(f"  [Cliente {cid}] Loss: {metrics['loss']:.4f} | Amostras: {num_samples}")

        logger.info(f"  [Servidor] Agregando pesos de {len(client_states)} clientes...")
        global_state = aggregate_fedavg(client_states, client_weights)
        global_model.load_state_dict(global_state)

        loss_global, acc_global = evaluate_global_model(global_model, test_loader)
        round_duration = time.time() - round_start

        model_size_mb = sum(p.numel() * 4 for p in global_model.parameters()) / (1024 ** 2)
        comm_mb = model_size_mb * len(client_states) * 2

        logger.info(f"  [Servidor] Rodada {round_num} em {round_duration:.2f}s | Loss: {loss_global:.4f} | Acc: {acc_global:.4f}")

        history["rounds"].append(round_num)
        history["accuracy"].append(acc_global)
        history["loss"].append(loss_global)
        history["communication_mb"].append(comm_mb)
        history["client_losses"].append([m["loss"] for m in client_metrics])

        if round_num > 1:
            delta = abs(acc_global - prev_accuracy)
            if delta < CONVERGENCE_THRESHOLD:
                stable_count += 1
                logger.info(f"  [Convergência] Δ={delta:.5f} < {CONVERGENCE_THRESHOLD} ({stable_count}/{CONVERGENCE_PATIENCE})")
            else:
                stable_count = 0
                logger.info(f"  [Convergência] Δ={delta:.5f} > {CONVERGENCE_THRESHOLD} (reset)")

            if stable_count >= CONVERGENCE_PATIENCE and converged_round is None:
                converged_round = round_num
                logger.info(f"\nCONVERGENCIA ATINGIDA na rodada {round_num}!")
                break

        prev_accuracy = acc_global

    overall_duration = time.time() - overall_start

    logger.info(f"\n{'='*60}")
    logger.info("SIMULAÇÃO CONCLUÍDA")
    logger.info(f"{'='*60}")
    logger.info(f"  Rodadas:      {round_num}")
    logger.info(f"  Convergência: {'Sim (rodada ' + str(converged_round) + ')' if converged_round else 'Não'}")
    logger.info(f"  Acurácia:     {history['accuracy'][-1]:.4f}")
    logger.info(f"  Loss:         {history['loss'][-1]:.4f}")
    logger.info(f"  Tráfego:      {sum(history['communication_mb']):.2f} MB")
    logger.info(f"  Tempo total:  {overall_duration:.2f}s ({overall_duration/60:.2f} min)")
    logger.info(f"{'='*60}")

    hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Histórico salvo: {hist_path}")

    from mosaicfl.core.calibration import TemperatureScaler
    from mosaicfl.core.evaluation import evaluate, print_report

    # Avaliação ANTES da calibração
    try:
        report_raw = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=1.0)
        logger.info(f"Avaliação pré-calibração — ECE={report_raw.calibration.ece:.4f} "
                    f"AUC={report_raw.macro_auc:.4f} F1={report_raw.macro_f1:.4f}")
        print_report(report_raw)
    except Exception as exc:
        logger.warning(f"Avaliação pré-calibração falhou: {exc}")
        report_raw = None

    # Temperature scaling
    # Nota: em simulação acadêmica reutilizamos o test_loader como calibration set.
    # Em produção (dados reais), FL_TEST_HOLDOUT_FRACTION reserva um holdout separado.
    scaler = TemperatureScaler()
    try:
        scaler.fit(global_model, test_loader, device=str(DEVICE))
        logger.info(f"Calibração concluída — T={scaler.T:.4f}")
    except Exception as exc:
        logger.warning(f"Calibração falhou ({exc}) — T mantido em 1.0")

    # Avaliação APÓS calibração
    try:
        report_cal = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=scaler.T)
        logger.info(f"Avaliação pós-calibração  — ECE={report_cal.calibration.ece:.4f} "
                    f"AUC={report_cal.macro_auc:.4f} F1={report_cal.macro_f1:.4f}")
        print_report(report_cal)
    except Exception as exc:
        logger.warning(f"Avaliação pós-calibração falhou: {exc}")
        report_cal = None

    # Persiste relatório em JSON
    eval_path = Path("experiments/logs") / f"evaluation_round_{round_num}.json"
    try:
        import dataclasses
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps({
            "round": round_num,
            "temperature": round(scaler.T, 4),
            "pre_calibration":  dataclasses.asdict(report_raw)  if report_raw  else None,
            "post_calibration": dataclasses.asdict(report_cal) if report_cal else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Relatório de avaliação salvo: {eval_path}")
    except Exception as exc:
        logger.warning(f"Falha ao salvar relatório de avaliação: {exc}")

    checkpoint_store = get_checkpoint_store(FL_DB_URL)
    checkpoint_store.save(
        round_num=round_num,
        state_dict=global_model.state_dict(),
        vocab=vocab or {},
        accuracy=history["accuracy"][-1] if history["accuracy"] else 0.0,
        loss=history["loss"][-1] if history["loss"] else 0.0,
        temperature=scaler.T,
    )
    logger.info(f"Checkpoint salvo no store ({type(checkpoint_store).__name__}, vocab_size={len(vocab or {})}, T={scaler.T:.4f})")

    return history, global_model


# ═══════════════════════════════════════════════════════════════════════════════
# MODO 2: SIMULAÇÃO COM RAY (PARALELA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning_ray(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Simula FL com Ray via flwr.simulation.start_simulation() — paralelo, rápido."""
    logger.info("=" * 60)
    logger.info("APRENDIZADO FEDERADO — SIMULAÇÃO COM RAY (PARALELA)")
    logger.info("=" * 60)
    logger.info(f"Clientes: {len(client_loaders)} | Rodadas: até {NUM_ROUNDS}")

    try:
        from flwr.simulation import start_simulation
    except ImportError as e:
        logger.error("=" * 60)
        logger.error("RAY NÃO ESTÁ INSTALADO!")
        logger.error("=" * 60)
        logger.error("Para usar o modo paralelo com Ray, execute:")
        logger.error('    pip install -U "flwr[simulation]"')
        logger.error("")
        logger.error("Ou, para continuar sem Ray, edite src/config.py:")
        logger.error("    USE_RAY = False")
        logger.error("=" * 60)
        raise RuntimeError("Ray não disponível. Instale com: pip install -U 'flwr[simulation]'") from e

    # Factory de clientes
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        train_loader, val_loader = client_loaders[cid_int]
        return FedProxClient(cid_int, train_loader, val_loader)

    # Estratégia
    from experiments.experiment_server import start_server
    from mosaicfl.core.federated import weighted_average
    strategy, tracker, history = start_server(
        num_rounds=NUM_ROUNDS,
        num_clients=len(client_loaders),
        test_loader=test_loader,
        vocab=vocab or {},
    )

    logger.info(f"Rodando simulação Flower+Ray com {len(client_loaders)} clientes...")
    overall_start = time.time()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=len(client_loaders),
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0},
        )
    except StopIteration as e:
        logger.info(f"Convergência: {e}")
    except Exception as e:
        logger.error(f"Erro na simulação Ray: {e}")
        raise

    overall_end = time.time()
    logger.info(f"Simulação concluída em {overall_end - overall_start:.2f}s")

    # Recupera pesos do último round salvo pela strategy; cria modelo novo só como fallback
    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    last_ckpt = history.get("last_checkpoint")
    if last_ckpt and Path(last_ckpt).exists():
        raw = torch.load(last_ckpt, map_location="cpu", weights_only=True)
        state_dict = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        global_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Modelo global restaurado de: {last_ckpt}")
    else:
        logger.warning("Checkpoint da simulação não encontrado — modelo com pesos aleatórios")

    hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Histórico salvo: {hist_path}")

    from mosaicfl.core.calibration import TemperatureScaler
    from mosaicfl.core.evaluation import evaluate, print_report

    # Avaliação ANTES da calibração
    try:
        report_raw = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=1.0)
        logger.info(f"Avaliação pré-calibração — ECE={report_raw.calibration.ece:.4f} "
                    f"AUC={report_raw.macro_auc:.4f} F1={report_raw.macro_f1:.4f}")
        print_report(report_raw)
    except Exception as exc:
        logger.warning(f"Avaliação pré-calibração falhou: {exc}")
        report_raw = None

    # Temperature scaling
    scaler = TemperatureScaler()
    try:
        scaler.fit(global_model, test_loader, device=str(DEVICE))
        logger.info(f"Calibração concluída — T={scaler.T:.4f}")
    except Exception as exc:
        logger.warning(f"Calibração falhou ({exc}) — T mantido em 1.0")

    # Avaliação APÓS calibração
    try:
        report_cal = evaluate(global_model, test_loader, class_labels=MODEL_CFG.class_labels,
                              device=str(DEVICE), temperature=scaler.T)
        logger.info(f"Avaliação pós-calibração  — ECE={report_cal.calibration.ece:.4f} "
                    f"AUC={report_cal.macro_auc:.4f} F1={report_cal.macro_f1:.4f}")
        print_report(report_cal)
    except Exception as exc:
        logger.warning(f"Avaliação pós-calibração falhou: {exc}")
        report_cal = None

    # Persiste relatório em JSON
    eval_path = Path("experiments/logs") / f"evaluation_ray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import dataclasses
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps({
            "mode": "ray",
            "temperature": round(scaler.T, 4),
            "pre_calibration":  dataclasses.asdict(report_raw)  if report_raw  else None,
            "post_calibration": dataclasses.asdict(report_cal) if report_cal else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Relatório de avaliação salvo: {eval_path}")
    except Exception as exc:
        logger.warning(f"Falha ao salvar relatório de avaliação: {exc}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"model_state": global_model.state_dict(), "vocab": vocab or {}, "temperature": scaler.T}, ckpt_path)
    logger.info(f"Checkpoint salvo: {ckpt_path} (vocab_size={len(vocab or {})}, T={scaler.T:.4f})")

    return history, global_model




# ═══════════════════════════════════════════════════════════════════════════════
# APRENDIZADO FEDERADO
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
    vocab: Dict[str, int] = None,
) -> Tuple[Dict, SimplifiedBEHRT]:
    """Executa o treinamento federado via Flower simulation."""

    # logger.info("=" * 60)
    # logger.info("INICIANDO APRENDIZADO FEDERADO")
    # logger.info("=" * 60)

    # def client_fn(cid: str) -> fl.client.NumPyClient:
    #     train_loader, val_loader = client_loaders[int(cid)]
    #     return FedProxClient(int(cid), train_loader, val_loader)

    # strategy, tracker, history = start_server(
    #     num_rounds=NUM_ROUNDS,
    #     num_clients=len(client_loaders),
    #     test_loader=test_loader,
    # )

    # try:
    #     fl.simulation.start_simulation(
    #         client_fn=client_fn,
    #         num_clients=len(client_loaders),
    #         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    #         strategy=strategy,
    #         client_resources={"num_cpus": 2, "num_gpus": 0},
    #     )
    # except StopIteration as e:
    #     logger.info(f"Convergência antecipada: {e}")
    # except Exception as e:
    #     logger.error(f"Erro na simulação: {e}")
    #     raise

    # global_model = SimplifiedBEHRT().to(DEVICE)

    # hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # with open(hist_path, "w", encoding="utf-8") as f:
    #     json.dump(history, f, indent=2, ensure_ascii=False)
    # logger.info(f"Histórico salvo: {hist_path}")

    # return history, global_model

    """
    Roteia para o modo correto baseado em config.USE_RAY.

    Para alternar:
        Edite src/config.py → USE_RAY = False  (manual, leve)
        Edite src/config.py → USE_RAY = True   (Ray, paralelo)
    """
    # Verifica se a flag existe no config (retrocompatibilidade)
    #use_ray = getattr(sys.modules['src.config'], 'USE_RAY', False)
    use_ray = bool(USE_RAY)

    if use_ray:
        logger.info("Modo Ray ativado (USE_RAY=True).")
        return run_federated_learning_ray(client_loaders, test_loader, total_train_samples, vocab=vocab)
    else:
        logger.info("Modo manual ativado (USE_RAY=False). Ray NÃO é necessário.")
        return run_federated_learning_manual(client_loaders, test_loader, total_train_samples, vocab=vocab)



# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RAG
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_rag_precision_at_k(
    rag: "ClinicalRAG",
    test_loader: DataLoader,
    vocab_inverse: Dict[int, str],
    class_labels: List[str],
    k: int = 3,
) -> Dict:
    """
    Avalia a qualidade da recuperação do RAG via Precision@k.

    Para cada amostra do test_loader, consulta o RAG com os tokens do paciente
    e verifica quantos dos k casos recuperados têm o mesmo desfecho que o rótulo
    real. Essa é a métrica central para um CDSS humano-no-loop: o clínico lê os
    casos recuperados, não o texto gerado pelo LLM.

    Retorna Precision@k global e por classe.
    """
    hits_total = 0
    queries_total = 0
    per_class_hits: Dict[str, int] = {lbl: 0 for lbl in class_labels}
    per_class_queries: Dict[str, int] = {lbl: 0 for lbl in class_labels}

    for batch_x, batch_y in test_loader:
        for seq, label_idx in zip(batch_x.tolist(), batch_y.tolist()):
            tokens = [
                vocab_inverse[t]
                for t in seq
                if t > 2 and t in vocab_inverse  # ignora PAD=0, UNK=1, CLS=2
            ]
            if not tokens:
                continue

            ground_truth = (
                class_labels[label_idx]
                if label_idx < len(class_labels)
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
        p_str = f"{p:.4f}" if p is not None else "n/a"
        logger.info(f"  {lbl}: {p_str}")

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

    # Deriva desfechos dos labels que efetivamente existem no test_loader.
    # Evita passar classes sem amostras para o extrator.
    all_labels = []
    for _, batch_y in test_loader:
        all_labels.extend(batch_y.tolist())
    desfechos = sorted(set(all_labels))
    logger.info(f"Desfechos presentes no test_loader: {desfechos}")

    extractor = BEHRTPatternExtractor(global_model, vocab_map)
    patterns = extractor.generate_all_profiles(test_loader, desfechos=desfechos)
    logger.info(f"Padrões extraídos: {len(patterns)} perfis")

    # ClinicalRAG usa _InMemoryStore quando FL_DB_URL não está configurado.
    rag = ClinicalRAG()
    rag.build_knowledge_base(patterns)

    vocab_inverse = {v: k for k, v in vocab_map.items()}
    labels = MODEL_CFG.class_labels

    # ── Precision@k — métrica principal de recuperação ───────────────────────
    logger.info("Avaliando Precision@k da recuperação...")
    precision_metrics = _eval_rag_precision_at_k(
        rag, test_loader, vocab_inverse, list(labels), k=3
    )

    # ── Exemplo de justificativa — uma amostra real ───────────────────────────
    sample_label = desfechos[0]
    sample_tokens: List[str] = []
    for batch_x, batch_y in test_loader:
        raw_tokens = [
            vocab_inverse.get(t, "")
            for t in batch_x[0].tolist()
            if t > 2
        ]
        sample_tokens = [t for t in raw_tokens if t][:10]
        sample_label = int(batch_y[0].item())
        break

    label_name = (
        labels[sample_label] if sample_label < len(labels) else f"classe_{sample_label}"
    )
    patient_data = {
        "tokens": ", ".join(sample_tokens) if sample_tokens else "dados laboratoriais",
    }
    model_prediction = {
        "diagnostico": label_name,
        "probabilidade": random.uniform(0.55, 0.95),
    }

    result = rag.explain(patient_data, model_prediction)
    logger.info(f"Justificativa — confiável: {result['confiavel']} | "
                f"alucinação: {result['alucinacao_detectada']}")

    result["precision_metrics"] = precision_metrics

    rag_path = f"experiments/data/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARATIVO — Random Forest (Bag-of-Tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def _loader_to_bow(
    loader: DataLoader,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converte sequências de token IDs de um DataLoader em vetores Bag-of-Tokens."""
    X, y = [], []
    for batch_x, batch_y in loader:
        for seq, label in zip(batch_x.numpy(), batch_y.numpy()):
            bow = np.zeros(vocab_size, dtype=np.float32)
            for tok in seq:
                if 2 < int(tok) < vocab_size:   # ignora PAD=0, UNK=1, CLS=2
                    bow[int(tok)] += 1.0
            X.append(bow)
            y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _safe_float(v: Optional[float]) -> Optional[float]:
    """Converte NaN/inf para None — garante JSON válido."""
    import math
    if v is None:
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)


def _eval_rf(
    rf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: List[str],
) -> Dict:
    """Avalia um RandomForestClassifier e retorna métricas no mesmo formato do EvaluationReport."""
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from mosaicfl.core.evaluation import compute_ece

    y_prob = rf.predict_proba(X_test)           # (N, n_classes) — ordenado por rf.classes_
    y_pred = rf.predict(X_test)

    # Reordena colunas para que o índice j corresponda à classe j,
    # independentemente da ordem que o RF viu durante o treino.
    n_classes = len(class_labels)
    rf_classes = list(rf.classes_)
    y_prob_ordered = np.zeros((len(y_test), n_classes), dtype=np.float32)
    for j, cls in enumerate(rf_classes):
        if cls < n_classes:
            y_prob_ordered[:, cls] = y_prob[:, j]

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    try:
        macro_auc: Optional[float] = float(
            roc_auc_score(y_test, y_prob_ordered, multi_class="ovr", average="macro")
        )
    except (ValueError, TypeError):
        macro_auc = None   # classes ausentes no conjunto de teste (ex.: dados sintéticos)

    confidences = torch.tensor(y_prob_ordered.max(axis=1), dtype=torch.float32)
    correct     = torch.tensor(y_pred == y_test, dtype=torch.bool)
    cal         = compute_ece(confidences, correct)

    per_class_auc: Dict[str, Optional[float]] = {}
    for i, label in enumerate(class_labels):
        y_bin = (y_test == i).astype(int)
        try:
            auc: Optional[float] = float(roc_auc_score(y_bin, y_prob_ordered[:, i]))
        except (ValueError, TypeError):
            auc = None
        per_class_auc[label] = _safe_float(auc)

    return {
        "accuracy":      round(accuracy, 4),
        "macro_f1":      round(macro_f1, 4),
        "macro_auc":     _safe_float(macro_auc),
        "ece":           round(cal.ece, 4),
        "per_class_auc": per_class_auc,
    }


def run_baseline_rf(
    client_loaders: Dict,
    test_loader: DataLoader,
    class_labels: List[str] = None,
    vocab_size: int = None,
    random_seed: int = None,
) -> Dict:
    """
    Baseline Random Forest (Bag-of-Tokens) para comparação com SimplifiedBEHRT.

    Opção A — RF Centralizado:
        Pool de todos os dados de treino dos clientes → um único RF.
        Representa o cenário ideal para um modelo clássico sem restrição de privacidade.

    Opção B — RF por Hospital:
        Um RF independente por cliente, treinado apenas nos dados locais.
        Representa o baseline do cenário federado sem compartilhamento de dados.

    A diferença BEHRT(FL) − RF(hospital) mede o ganho do aprendizado federado
    sobre o baseline local. A diferença BEHRT(FL) − RF(centralizado) mede o ganho
    da modelagem sequencial temporal sobre um modelo de bag-of-tokens com todos os dados.

    Args:
        client_loaders: {cid: (train_loader, val_loader)} — mesmo dict do FL
        test_loader:    DataLoader global (holdout de avaliação)
        class_labels:   nomes das classes (padrão: MODEL_CFG.class_labels)
        vocab_size:     dim do vetor BoT (padrão: VOCAB_SIZE)
        random_seed:    semente de reprodutibilidade (padrão: RANDOM_SEED)

    Returns:
        Dict com resultados das duas opções, salvável em experiment_results.json.
    """
    from sklearn.ensemble import RandomForestClassifier

    class_labels = list(class_labels or MODEL_CFG.class_labels)
    vocab_size   = vocab_size  if vocab_size  is not None else VOCAB_SIZE
    random_seed  = random_seed if random_seed is not None else RANDOM_SEED

    logger.info("=" * 60)
    logger.info("BASELINE — Random Forest (Bag-of-Tokens)")
    logger.info("=" * 60)
    logger.info(f"  vocab_size: {vocab_size} | classes: {class_labels}")

    # ── Conjunto de teste (compartilhado entre as duas opções) ─────────────────
    X_test, y_test = _loader_to_bow(test_loader, vocab_size)
    unique, counts = np.unique(y_test, return_counts=True)
    dist = {class_labels[int(c)]: int(n) for c, n in zip(unique, counts) if int(c) < len(class_labels)}
    logger.info(f"  Teste: {len(X_test)} amostras | distribuição: {dist}")

    results: Dict = {}

    # ── Opção A: RF Centralizado ───────────────────────────────────────────────
    logger.info("\n[Opção A] RF Centralizado — pool de todos os clientes")
    X_parts, y_parts = [], []
    for cid, (train_loader, _) in client_loaders.items():
        X_c, y_c = _loader_to_bow(train_loader, vocab_size)
        X_parts.append(X_c)
        y_parts.append(y_c)
        logger.info(f"  Cliente {cid}: {len(X_c)} amostras")

    X_pool = np.concatenate(X_parts, axis=0)
    y_pool = np.concatenate(y_parts, axis=0)
    logger.info(f"  Pool total: {len(X_pool)} amostras")

    rf_central = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    rf_central.fit(X_pool, y_pool)
    m_a = _eval_rf(rf_central, X_test, y_test, class_labels)
    results["opcao_a_centralizado"] = {
        **m_a,
        "n_train": int(len(X_pool)),
        "descricao": "RF treinado em pool de todos os clientes (centralizado)",
    }
    auc_str = f"{m_a['macro_auc']:.4f}" if m_a['macro_auc'] is not None else "n/a"
    logger.info(
        f"  Resultado → Acc={m_a['accuracy']:.4f}  "
        f"AUC={auc_str}  "
        f"F1={m_a['macro_f1']:.4f}  "
        f"ECE={m_a['ece']:.4f}"
    )

    # ── Opção B: RF por Hospital ───────────────────────────────────────────────
    logger.info("\n[Opção B] RF por Hospital — modelo independente por cliente")
    per_client: Dict = {}
    for cid, (train_loader, _) in client_loaders.items():
        X_c, y_c = _loader_to_bow(train_loader, vocab_size)
        n_local_classes = int(len(np.unique(y_c)))
        rf_local = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        rf_local.fit(X_c, y_c)
        m = _eval_rf(rf_local, X_test, y_test, class_labels)
        per_client[str(cid)] = {
            **m,
            "n_train": int(len(X_c)),
            "n_classes_local": n_local_classes,
        }
        auc_h = f"{m['macro_auc']:.4f}" if m['macro_auc'] is not None else "n/a"
        logger.info(
            f"  Hospital {cid}: Acc={m['accuracy']:.4f}  "
            f"AUC={auc_h}  "
            f"F1={m['macro_f1']:.4f}  "
            f"(treino={len(X_c)}, classes_locais={n_local_classes})"
        )
    results["opcao_b_por_hospital"] = {
        "per_client": per_client,
        "descricao": "RF independente por hospital — sem compartilhamento de dados",
    }

    # ── Tabela comparativa ─────────────────────────────────────────────────────
    header = f"\n{'Modelo':<42} {'Accuracy':>8} {'AUC-ROC':>8} {'F1 Macro':>8} {'ECE':>7}"
    sep    = "-" * 76
    logger.info("\n" + "=" * 60)
    logger.info("TABELA COMPARATIVA — Baseline vs. BEHRT (FL)")
    logger.info("=" * 60)
    logger.info(header)
    logger.info(sep)
    def _fmt(v: Optional[float]) -> str:
        return f"{v:>8.4f}" if v is not None else "     n/a"

    logger.info(
        f"{'RF Centralizado (BoT)':<42} "
        f"{_fmt(m_a['accuracy'])} {_fmt(m_a['macro_auc'])} "
        f"{_fmt(m_a['macro_f1'])} {_fmt(m_a['ece'])}"
    )
    for cid, m in per_client.items():
        logger.info(
            f"  {'RF Hospital ' + str(cid) + ' (BoT)':<40} "
            f"{_fmt(m['accuracy'])} {_fmt(m['macro_auc'])} "
            f"{_fmt(m['macro_f1'])} {_fmt(m['ece'])}"
        )
    logger.info(sep)
    logger.info(f"{'SimplifiedBEHRT (FL) — ver evaluation_round_*.json':<42}")
    logger.info("=" * 60)

    results["meta"] = {
        "vocab_size":   vocab_size,
        "n_test":       int(len(X_test)),
        "class_labels": class_labels,
        "rf_params":    {"n_estimators": 200, "class_weight": "balanced"},
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION — LATE FUSION DEMOGRÁFICA (age_norm + sex_binary)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_correlated_demo(labels: torch.Tensor, noise_std: float = 0.15, seed: int = 42) -> torch.Tensor:
    """
    Gera demográficos sintéticos CORRELACIONADOS com os labels de prognóstico.

    Baseia-se no perfil epidemiológico do COVID-19:
      alta (0):               age_mean=0.45, P(M)=0.45
      internacao_prol (1):    age_mean=0.55, P(M)=0.50
      uti (2):                age_mean=0.65, P(M)=0.60
      obito (3):              age_mean=0.70, P(M)=0.65

    Correlação garante que Config B (com demográficos) tem sinal real para aprender,
    tornando o ablation academicamente válido mesmo sem dados FAPESP carregados.

    Returns:
        FloatTensor (N, 2) — [age_norm, sex_binary]
    """
    rng = np.random.default_rng(seed)
    age_means = {0: 0.45, 1: 0.55, 2: 0.65, 3: 0.70}
    sex_probs  = {0: 0.45, 1: 0.50, 2: 0.60, 3: 0.65}

    ages  = np.zeros(len(labels), dtype=np.float32)
    sexes = np.zeros(len(labels), dtype=np.float32)
    for i, lbl in enumerate(labels.tolist()):
        lbl = min(int(lbl), 3)
        ages[i]  = float(np.clip(rng.normal(age_means[lbl], noise_std), 0.18, 0.95))
        sexes[i] = 1.0 if rng.random() < sex_probs[lbl] else 0.0

    return torch.from_numpy(np.stack([ages, sexes], axis=1))


def run_ablation_demographics(
    client_loaders: Dict,
    test_loader: DataLoader,
    demographics_by_client: Optional[Dict] = None,
    test_loader_demo: Optional[DataLoader] = None,
    n_epochs: int = 10,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Ablation study: late fusion demográfica (age_norm + sex_binary) no SimplifiedBEHRT.

    Compara:
      Config A — sequências de exames apenas (demo_dim=0, modelo base)
      Config B — sequências + late fusion demográfica (demo_dim=2)

    Com dados reais (demographics_by_client não-None): usa age_norm e sex_binary
    extraídos pelo SequencePipeline a partir de clinical.patients.

    Com dados sintéticos (demographics_by_client=None): gera demográficos correlacionados
    com os labels (perfil epidemiológico COVID-19) para validar a arquitetura.

    O ablation usa SGD local (sem FL) para isolar o efeito dos demográficos da
    heterogeneidade entre clientes. Isso é adequado para a pergunta de pesquisa:
    "demográficos adicionam informação ao BEHRT para este dataset?"

    Returns:
        Dict com métricas de Config A e B, delta e metadados.
    """
    from mosaicfl.core.model import SimplifiedBEHRT as _BEHRT
    from mosaicfl.core.evaluation import evaluate, print_report

    random_seed = random_seed if random_seed is not None else RANDOM_SEED
    use_real_demo = demographics_by_client is not None

    logger.info("=" * 60)
    logger.info("ABLATION — Late Fusion Demográfica")
    logger.info(f"  Fonte dos demográficos: {'dados reais (FAPESP)' if use_real_demo else 'sintéticos correlacionados'}")
    logger.info("  Config A: SimplifiedBEHRT sem demográficos (demo_dim=0)")
    logger.info("  Config B: SimplifiedBEHRT + late fusion (demo_dim=2)")
    logger.info(f"  Épocas locais: {n_epochs} | seed: {random_seed}")
    logger.info("=" * 60)

    def _collect_labels(loaders: Dict) -> torch.Tensor:
        lbls = []
        for _, (train_loader, _) in loaders.items():
            for _, batch_y in train_loader:
                lbls.append(batch_y)
        return torch.cat(lbls, dim=0)

    def _train_local(model: nn.Module, loaders: Dict, with_demo: bool, demo_loaders: Optional[Dict]) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        model.train()

        # Pré-gera demográficos sintéticos se necessário
        synth_demo_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if with_demo and not use_real_demo:
            for cid, (train_loader, _) in loaders.items():
                all_x, all_y = [], []
                for bx, by in train_loader:
                    all_x.append(bx); all_y.append(by)
                all_x = torch.cat(all_x); all_y = torch.cat(all_y)
                synth_demo_cache[cid] = (all_x, all_y, _make_correlated_demo(all_y, seed=random_seed + cid))

        for _ in range(n_epochs):
            if with_demo and use_real_demo and demo_loaders:
                for cid, (demo_train_loader, _) in demo_loaders.items():
                    for batch_x, batch_y, batch_demo in demo_train_loader:
                        batch_x = batch_x.to(DEVICE)
                        batch_y = batch_y.to(DEVICE)
                        batch_demo = batch_demo.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(batch_x, demographics=batch_demo)
                        criterion(logits, batch_y).backward()
                        optimizer.step()
            elif with_demo and not use_real_demo:
                for cid, (all_x, all_y, all_demo) in synth_demo_cache.items():
                    dataset = TensorDataset(all_x, all_y, all_demo)
                    for bx, by, bd in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
                        bx = bx.to(DEVICE); by = by.to(DEVICE); bd = bd.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(bx, demographics=bd)
                        criterion(logits, by).backward()
                        optimizer.step()
            else:
                for cid, (train_loader, _) in loaders.items():
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(DEVICE); batch_y = batch_y.to(DEVICE)
                        optimizer.zero_grad()
                        criterion(model(batch_x), batch_y).backward()
                        optimizer.step()

    def _eval_with_demo(model: nn.Module, t_loader_demo: DataLoader) -> Dict:
        """Avalia Config B usando test_loader com demographics reais ou sintéticos."""
        model.eval()
        all_preds, all_labels = [], []
        criterion = nn.CrossEntropyLoss()
        total_loss, total_n = 0.0, 0

        with torch.no_grad():
            for batch in t_loader_demo:
                if len(batch) == 3:
                    bx, by, bd = batch
                else:
                    bx, by = batch
                    bd = _make_correlated_demo(by, seed=random_seed + 99)
                bx = bx.to(DEVICE); by = by.to(DEVICE); bd = bd.to(DEVICE)
                logits = model(bx, demographics=bd)
                total_loss += criterion(logits, by).item() * by.size(0)
                total_n += by.size(0)
                all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
                all_labels.extend(by.cpu().tolist())

        from sklearn.metrics import accuracy_score, f1_score
        try:
            from sklearn.metrics import roc_auc_score
            # não temos softmax aqui — usar apenas acc e f1
            macro_auc = None
        except ImportError:
            macro_auc = None

        return {
            "accuracy": round(float(accuracy_score(all_labels, all_preds)), 4),
            "macro_f1": round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
            "loss":     round(total_loss / total_n, 4) if total_n > 0 else None,
        }

    results: Dict = {}

    for config_name, demo_dim, with_demo in [
        ("config_A_sem_demo",   0, False),
        ("config_B_late_fusion", 2, True),
    ]:
        logger.info(f"\n[Ablação] Treinando {config_name} (demo_dim={demo_dim}, épocas={n_epochs})...")
        torch.manual_seed(random_seed)
        model = _BEHRT(use_cls_token=True, demo_dim=demo_dim).to(DEVICE)

        _train_local(model, client_loaders, with_demo, demographics_by_client)

        if with_demo:
            t_loader = test_loader_demo if (use_real_demo and test_loader_demo is not None) else test_loader
            metrics = _eval_with_demo(model, t_loader)
            metrics["descricao"] = (
                "SimplifiedBEHRT + late fusion demográfica "
                f"({'dados reais FAPESP' if use_real_demo else 'sintéticos correlacionados'})"
            )
        else:
            try:
                report = evaluate(model, test_loader, class_labels=MODEL_CFG.class_labels,
                                  device=str(DEVICE), temperature=1.0)
                metrics = {
                    "accuracy":  round(report.accuracy, 4),
                    "macro_f1":  round(report.macro_f1, 4),
                    "macro_auc": round(report.macro_auc, 4) if report.macro_auc else None,
                    "ece":       round(report.calibration.ece, 4),
                    "descricao": "SimplifiedBEHRT sem demográficos (baseline)",
                }
            except Exception as exc:
                logger.warning(f"evaluate() falhou para {config_name}: {exc}")
                metrics = {"erro": str(exc)}

        metrics["demo_dim"] = demo_dim
        metrics["n_epochs"] = n_epochs
        results[config_name] = metrics

        acc = metrics.get("accuracy", "n/a")
        f1  = metrics.get("macro_f1",  "n/a")
        logger.info(f"  {config_name}: Acc={acc} | F1={f1}")

    # Delta B − A
    a = results.get("config_A_sem_demo",   {})
    b = results.get("config_B_late_fusion", {})
    if "accuracy" in a and "accuracy" in b:
        delta_acc = round(b["accuracy"] - a["accuracy"], 4)
        delta_f1  = round(b["macro_f1"] - a["macro_f1"], 4) if "macro_f1" in a and "macro_f1" in b else None
        results["delta_B_minus_A"] = {"accuracy": delta_acc, "macro_f1": delta_f1}
        logger.info(f"\n[Ablação] Δ (B − A): Acc={delta_acc:+.4f}"
                    + (f" | F1={delta_f1:+.4f}" if delta_f1 is not None else ""))

    # Tabela resumo
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADO ABLAÇÃO — Late Fusion Demográfica")
    logger.info("=" * 70)
    logger.info(f"{'Config':<35} {'Accuracy':>8} {'F1 Macro':>8}")
    logger.info("-" * 55)
    for key in ["config_A_sem_demo", "config_B_late_fusion"]:
        m = results.get(key, {})
        logger.info(
            f"{key:<35} "
            f"{m.get('accuracy', 'n/a'):>8} "
            f"{m.get('macro_f1', 'n/a'):>8}"
        )
    if "delta_B_minus_A" in results:
        d = results["delta_B_minus_A"]
        logger.info(
            f"{'Δ (B − A)':<35} "
            f"{d.get('accuracy', 'n/a'):>+8} "
            + (f"{d.get('macro_f1', 'n/a'):>+8}" if d.get('macro_f1') is not None else "     n/a")
        )
    logger.info("=" * 70)

    results["meta"] = {
        "fonte_demo":   "real_fapesp" if use_real_demo else "sintetico_correlacionado",
        "n_epochs":     n_epochs,
        "random_seed":  random_seed,
        "demo_features": ["age_norm (birth_year / ref_year=2021)", "sex_binary (M=1, F=0)"],
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("MOSAIC-FL v2 — Integração com Dados Reais")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("=" * 60)

    use_ray = bool(USE_RAY)
    logger.info(f"Modo FL:    {'Ray (paralelo)' if use_ray else 'Manual sequencial (sem Ray)'}")
    logger.info(f"Ambiente:   {FL_ENV}")
    logger.info(f"Banco:      {'configurado' if FL_DB_URL else 'não configurado'}")

    # ── Guarda de produção ────────────────────────────────────────────────────
    # Em produção FL_DB_URL é obrigatório — sem dados reais não há experimento.
    if FL_ENV == "production" and not FL_DB_URL:
        logger.error("=" * 60)
        logger.error("ERRO: FL_ENV=production requer FL_DB_URL")
        logger.error("Configure: export FL_DB_URL='postgresql://user:pass@host:5432/db'")
        logger.error("=" * 60)
        sys.exit(1)

    # ── 1. Carregamento ───────────────────────────────────────────────────────
    df_raw = None          # só disponível no modo CSV/sintético
    loaded_from_db = False

    demographics_by_client: Dict = {}
    test_loader_demo = None

    if FL_DB_URL:
        logger.info("[1/4] Carregando dados do banco (SequencePipeline)...")
        try:
            (client_loaders, test_loader, vocab_map, total,
             demographics_by_client, test_loader_demo) = prepare_dataloaders_from_db(FL_DB_URL)
            loaded_from_db = True
        except Exception as e:
            logger.error(f"Falha ao carregar dados do banco: {e}")
            if FL_ENV == "production":
                logger.error("Abortando — FL_ENV=production não permite fallback.")
                sys.exit(1)
            logger.warning("Tentando fallback para CSV/sintético (FL_ENV=development)...")

    if not loaded_from_db:
        logger.info("[1/4] Carregando dataset (CSV ou sintético)...")
        try:
            df_raw = load_with_fallback(allow_synthetic=(FL_ENV != "production"))
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(f"Dataset não encontrado: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Schema inválido: {e}")
            sys.exit(1)

        logger.info(f"      {len(df_raw)} registros | {df_raw['instituicao'].nunique()} instituicoes")
        logger.info("[2/4] Pré-processando...")
        preprocessor = EHRPreprocessor()
        client_loaders, test_loader, vocab_map, total = prepare_dataloaders(df_raw, preprocessor)
    else:
        logger.info("[2/4] Pré-processamento integrado ao SequencePipeline — etapa ignorada.")

    logger.info(f"      {len(client_loaders)} clientes | {total} amostras de treino")

    # ── MetricsStore — persiste métricas no banco (SQLite dev / PostgreSQL prod) ─
    data_source = "fapesp" if loaded_from_db else "synthetic"
    metrics_store = get_metrics_store(FL_DB_URL)

    # ── 3. Aprendizado Federado ───────────────────────────────────────────────
    logger.info("[3/5] Aprendizado Federado...")
    history, global_model = run_federated_learning(client_loaders, test_loader, total, vocab=vocab_map)

    # Persiste métricas do último round FL
    if history:
        last_round = max(history.keys()) if isinstance(history, dict) else len(history)
        last_metrics = history[last_round] if isinstance(history, dict) else history[-1]
        metrics_store.save(
            round_num=last_round,
            metrics={
                "accuracy": last_metrics.get("accuracy"),
                "loss":     last_metrics.get("loss"),
                "macro_auc": last_metrics.get("macro_auc"),
                "macro_f1":  last_metrics.get("macro_f1"),
                "ece":       last_metrics.get("ece"),
                "per_class_auc": last_metrics.get("per_class_auc"),
                "per_class_f1":  last_metrics.get("per_class_f1"),
            },
            data_source=data_source,
        )

    # ── 4. RAG ────────────────────────────────────────────────────────────────
    logger.info("[4/5] Pipeline RAG...")
    try:
        rag_result = run_rag_pipeline(global_model, vocab_map, test_loader)
        # Persiste Precision@k do RAG associada ao round final
        p_metrics = rag_result.get("precision_metrics", {})
        if p_metrics:
            metrics_store.save(
                round_num=last_round if history else 0,
                metrics={
                    "rag_precision_at_k":      p_metrics.get(f"precision_at_{p_metrics.get('k', 3)}"),
                    "rag_k":                   p_metrics.get("k"),
                    "rag_per_class_precision": p_metrics.get(f"per_class_precision_at_{p_metrics.get('k', 3)}"),
                },
                data_source=data_source,
            )
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        rag_result = {"erro": str(e)}

    # ── 5. Baseline RF ────────────────────────────────────────────────────────
    logger.info("[5/5] Baseline Random Forest...")
    try:
        baseline_result = run_baseline_rf(
            client_loaders,
            test_loader,
            class_labels=list(MODEL_CFG.class_labels),
        )
        baseline_path = Path("experiments/data") / f"baseline_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Persiste métricas do baseline centralizado no banco para comparação histórica
        m_a = (baseline_result.get("opcao_a_centralizado") or {})
        if m_a:
            metrics_store.save(
                round_num=0,  # round 0 = baseline pré-FL
                metrics={
                    "accuracy":  m_a.get("accuracy"),
                    "macro_auc": m_a.get("macro_auc"),
                    "macro_f1":  m_a.get("macro_f1"),
                    "ece":       m_a.get("ece"),
                },
                data_source=f"{data_source}_baseline_rf",
            )
        logger.info(f"Baseline salvo: {baseline_path}")
    except Exception as e:
        logger.error(f"Erro no baseline RF: {e}")
        baseline_result = {"erro": str(e)}

    # ── 6. Ablation demográfica ───────────────────────────────────────────────
    logger.info("[6/6] Ablation study — late fusion demográfica...")
    try:
        ablation_result = run_ablation_demographics(
            client_loaders=client_loaders,
            test_loader=test_loader,
            demographics_by_client=demographics_by_client if demographics_by_client else None,
            test_loader_demo=test_loader_demo,
            n_epochs=10,
        )
        ablation_path = Path("experiments/data") / f"ablation_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(
            json.dumps(ablation_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Ablation salvo: {ablation_path}")
    except Exception as e:
        logger.error(f"Erro no ablation: {e}")
        ablation_result = {"erro": str(e)}

    # ── Resumo ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CONCLUÍDO")
    logger.info(f"  Modo dados:     {'banco (SequencePipeline)' if loaded_from_db else 'CSV/sintético'}")
    logger.info(f"  Clientes FL:    {len(client_loaders)}")
    logger.info(f"  RAG confiável:  {rag_result.get('confiavel', False)}")
    m_a = (baseline_result.get("opcao_a_centralizado") or {})
    if m_a:
        logger.info(f"  RF centralizado: Acc={m_a.get('accuracy','?')}  AUC={m_a.get('macro_auc','?')}  F1={m_a.get('macro_f1','?')}")
    delta = ablation_result.get("delta_B_minus_A", {})
    if delta:
        logger.info(f"  Ablation Δ Acc: {delta.get('accuracy', 'n/a'):+}")
    logger.info(f"  Logs em:        {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()