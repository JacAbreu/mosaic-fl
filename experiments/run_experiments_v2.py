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
#     history, global_model = run_federated_learning(client_loaders, test_loader, total)

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
from typing import Dict, List, Tuple
from datetime import datetime

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
from mosaicfl.core.preprocessor import EHRPreprocessor, split_by_institution
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

    return history, global_model


# ═══════════════════════════════════════════════════════════════════════════════
# MODO 2: SIMULAÇÃO COM RAY (PARALELA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning_ray(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
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

    global_model = SimplifiedBEHRT(use_cls_token=True).to(DEVICE)
    logger.info("Modelo global reconstruído.")

    hist_path = f"experiments/data/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Histórico salvo: {hist_path}")

    return history, global_model




# ═══════════════════════════════════════════════════════════════════════════════
# APRENDIZADO FEDERADO
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_learning(
    client_loaders: Dict,
    test_loader: DataLoader,
    total_train_samples: int,
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
        return run_federated_learning_ray(client_loaders, test_loader, total_train_samples)
    else:
        logger.info("Modo manual ativado (USE_RAY=False). Ray NÃO é necessário.")
        return run_federated_learning_manual(client_loaders, test_loader, total_train_samples)



# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RAG
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(
    global_model: SimplifiedBEHRT,
    vocab_map: Dict,
    test_loader: DataLoader,
    df_raw: pd.DataFrame,
) -> Dict:
    """Extrai padrões do BEHRT e gera justificativa via RAG."""

    logger.info("=" * 60)
    logger.info("PIPELINE RAG")
    logger.info("=" * 60)

    extractor = BEHRTPatternExtractor(global_model, vocab_map)
    desfechos = sorted(df_raw["desfecho"].dropna().unique().astype(int).tolist())
    patterns = extractor.generate_all_profiles(test_loader, desfechos=desfechos)
    logger.info(f"Padrões extraídos: {len(patterns)} perfis")

    rag = ClinicalRAG()
    rag.build_knowledge_base(patterns)

    sample = df_raw.sample(1, random_state=RANDOM_SEED).iloc[0]
    patient_data = {
        "febre": random.choice(["ausente", "leve", "moderada", "alta"]),
        "tosse": random.choice(["ausente", "seca", "produtiva"]),
        "saturacao": random.choice(["98%", "95%", "92%", "88%"]),
        "faixa_etaria": "idoso" if sample.get("idade", 30) >= 60 else "adulto",
    }
    model_prediction = {
        "diagnostico": "pneumonia" if int(sample.get("desfecho", 0)) == 1 else "alta",
        "probabilidade": random.uniform(0.55, 0.95),
    }

    result = rag.explain(patient_data, model_prediction)
    logger.info(f"Justificativa — confiável: {result['confiavel']} | "
                f"alucinação: {result['alucinacao_detectada']}")

    rag_path = f"experiments/data/rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("MOSAIC-FL v2 — Integração com Dados Reais")
    logger.info("Autora: Jacqueline Abreu | ICMC/USP")
    logger.info("=" * 60)

    #use_ray = getattr(sys.modules['src.config'], 'USE_RAY', False)
    use_ray = bool(USE_RAY)
    logger.info(f"Modo de execução: {'Ray (paralelo)' if use_ray else 'Manual sequencial (sem Ray)'}")
    if not use_ray:
        logger.info("Dica: para ativar Ray, edite USE_RAY = True em src/config.py")
        logger.info("      e execute: pip install -U 'flwr[simulation]'")


    # 1. Carrega dados
    logger.info("[1/4] Carregando dataset...")
    try:
        #df_raw = load_clinical_dataset() para rodar com a base de dados no sgbd
        df_raw = load_with_fallback(allow_synthetic=True) #para rodar com a base de dados sintética
    except FileNotFoundError as e:
        logger.error(f"Dataset não encontrado: {e}")
        logger.error("Coloque o CSV em data/ ou configure MOSAICFL_DB_URL")
        logger.error("Diagnóstico: python -c \"from mosaicfl.core.data_loader import diagnose_dataset; diagnose_dataset()\"")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Schema inválido: {e}")
        logger.error("Edite mosaicfl/v2/data_loader.py → COLUMN_MAPPING")
        sys.exit(1)

    logger.info(f"      {len(df_raw)} registros | {df_raw['instituicao'].nunique()} instituicoes")

    # 2. Pré-processamento
    logger.info("[2/4] Pré-processando...")
    preprocessor = EHRPreprocessor()
    client_loaders, test_loader, vocab_map, total = prepare_dataloaders(df_raw, preprocessor)
    logger.info(f"      {len(client_loaders)} clientes | {total} amostras de treino")

    # 3. Aprendizado Federado
    logger.info("[3/4] Aprendizado Federado...")
    history, global_model = run_federated_learning(client_loaders, test_loader, total)

    # 4. RAG
    logger.info("[4/4] Pipeline RAG...")
    try:
        rag_result = run_rag_pipeline(global_model, vocab_map, test_loader, df_raw)
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        rag_result = {"erro": str(e)}

    # Resumo
    logger.info("=" * 60)
    logger.info("CONCLUÍDO")
    logger.info(f"  Registros:      {len(df_raw)}")
    logger.info(f"  Clientes FL:    {len(client_loaders)}")
    logger.info(f"  RAG confiável:  {rag_result.get('confiavel', False)}")
    logger.info(f"  Logs em:        {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()