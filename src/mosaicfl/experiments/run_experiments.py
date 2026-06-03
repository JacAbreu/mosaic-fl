"""
ATENÇÃO — arquivo legado, mantido apenas para compatibilidade.

Este arquivo era o orquestrador original dos experimentos e foi substituído
por runner.py, que contém todas as correções aplicadas:

  - Exp2/3: seed fixo (RANDOM_SEED) e curvas derivadas dos dados reais
  - Exp5:   avaliação global real via get_evaluate_fn (server.py)
  - Exp4:   chave 'alucinacao_detectada' corrigida (rag_system.py)
  - Imports atualizados para o pacote instalado (mosaicfl.*)

NÃO edite este arquivo. Use runner.py para qualquer alteração.
"""
import warnings

warnings.warn(
    "\n\n"
    "  run_experiments.py está obsoleto.\n"
    "  Execute runner.py diretamente:\n"
    "      python -m mosaicfl.experiments.runner\n"
    "  ou via makefile:\n"
    "      make run\n",
    DeprecationWarning,
    stacklevel=2,
)

# Redireciona para o runner atualizado
from mosaicfl.experiments.runner import main

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# CÓDIGO ORIGINAL (legado) — mantido como referência histórica
# ---------------------------------------------------------------------------
# Orquestrador dos 5 experimentos do TCC.
# Experimento 1: Padronização | 2: Efeito Equalizador | 3: Heterogeneidade não-IID
# Experimento 4: RAG | 5: Eficiência Operacional
# import torch
# import numpy as np
# import pandas as pd
# import json
# import time
# from sklearn.metrics import roc_auc_score, accuracy_score
# from torch.utils.data import DataLoader, TensorDataset
# 
# from preprocess import EHRPreprocessor, split_by_institution
# from model import SimplifiedBEHRT
# from client import create_client_fn
# from server import start_server, ConvergenceTracker
# from rag_system import ClinicalRAG
# from config import *
# 
# 
# class ExperimentRunner:
#     def __init__(self):
#         self.results = {}
# 
#     def exp1_standardization(self, df_raw: pd.DataFrame):
#         """Experimento 1: Mapear desafios de padronização."""
#         print("=" * 60)
#         print("EXPERIMENTO 1: Padronização e Pré-processamento")
#         print("=" * 60)
# 
#         preprocessor = EHRPreprocessor()
#         start_time = time.time()
#         df_processed, summary = preprocessor.process(df_raw, text_cols=['sintoma', 'exame'])
#         vocab = preprocessor.vocab_map
#         elapsed = time.time() - start_time
# 
#         summary["tempo_segundos"] = elapsed
#         self.results["exp1"] = summary
# 
#         print(f"\nResumo: {json.dumps(summary, indent=2, ensure_ascii=False)}")
#         return df_processed
# 
#     def exp2_equalizing_effect(self, clients_data: Dict[int, pd.DataFrame]):
#         """Experimento 2: Comparar Local vs. Federado (AUC-ROC)."""
#         print("\n" + "=" * 60)
#         print("EXPERIMENTO 2: Efeito Equalizador do FL")
#         print("=" * 60)
# 
#         local_aucs = {}
#         federated_aucs = {}
# 
#         # Treinamento local isolado (baseline)
#         for cid, df in clients_data.items():
#             print(f"\nTreinamento LOCAL - Cliente {cid} ({len(df)} amostras)")
#             # Aqui você treinaria o modelo localmente e mediria AUC-ROC
#             local_aucs[cid] = np.random.uniform(0.65, 0.75)  # placeholder
# 
#         # Treinamento federado (simulado)
#         print(f"\nTreinamento FEDERADO - FedProx (mu={PROXIMAL_MU})")
#         # Aqui você executaria o Flower server + clients
#         for cid in clients_data.keys():
#             federated_aucs[cid] = np.random.uniform(0.78, 0.88)  # placeholder
# 
#         # Análise: ganho do menor cliente
#         smallest_client = min(clients_data.keys(), key=lambda k: len(clients_data[k]))
#         gain = federated_aucs[smallest_client] - local_aucs[smallest_client]
# 
#         self.results["exp2"] = {
#             "local_aucs": local_aucs,
#             "federated_aucs": federated_aucs,
#             "ganho_menor_cliente": gain,
#             "menor_cliente": smallest_client
#         }
#         print(f"Ganho AUC-ROC do menor cliente (HF{smallest_client+1}): +{gain:.3f}")
# 
#     def exp3_non_iid_impact(self, clients_data: Dict[int, pd.DataFrame]):
#         """Experimento 3: Impacto da heterogeneidade não-IID."""
#         print("\n" + "=" * 60)
#         print("EXPERIMENTO 3: Impacto da Heterogeneidade (não-IID)")
#         print("=" * 60)
# 
#         # Análise demográfica por cliente
#         demographics = {}
#         for cid, df in clients_data.items():
#             if 'idade' in df.columns:
#                 demographics[cid] = {
#                     "media_idade": df['idade'].mean(),
#                     "std_idade": df['idade'].std(),
#                     "pct_idoso": (df['idade'] > 60).mean() * 100
#                 }
# 
#         # Simulação: acurácia por subgrupo ao longo das rodadas
#         rounds = list(range(0, NUM_ROUNDS + 1, 5))
#         acc_idosos = [0.60 + 0.25 * (1 - np.exp(-0.1 * r)) for r in rounds]
#         acc_criancas = [0.55 + 0.20 * (1 - np.exp(-0.08 * r)) for r in rounds]
# 
#         self.results["exp3"] = {
#             "demografia": demographics,
#             "convergencia_idosos": dict(zip(rounds, acc_idosos)),
#             "convergencia_criancas": dict(zip(rounds, acc_criancas)),
#             "gap_final": abs(acc_idosos[-1] - acc_criancas[-1])
#         }
#         print(f"Gap final de acurácia (idosos vs. crianças): {abs(acc_idosos[-1] - acc_criancas[-1]):.3f}")
# 
#     def exp4_rag_uncertainty(self, rag: ClinicalRAG, test_samples: List[Dict]):
#         """Experimento 4: Avaliação qualitativa do RAG (escala Likert 1-5)."""
#         print("\n" + "=" * 60)
#         print("EXPERIMENTO 4: RAG na Redução de Incertezas")
#         print("=" * 60)
# 
#         scores = []
#         hallucinations = 0
# 
#         for i, sample in enumerate(test_samples):
#             result = rag.explain(sample, {
#                 "diagnostico": sample.get("desfecho_previsto", "pneumonia"),
#                 "probabilidade": sample.get("prob", 0.87)
#             })
# 
#             # Simulação de avaliação Likert (1-5)
#             score = np.random.choice([4, 5], p=[0.3, 0.7]) if not result['alucinacao'] else np.random.choice([1, 2, 3])
#             scores.append(score)
#             if result['alucinacao']:
#                 hallucinations += 1
# 
#             print(f"Amostra {i+1}: Score={score} | Alucinação={result['alucinacao']}")
# 
#         useful_pct = (np.array(scores) >= 4).mean() * 100
#         self.results["exp4"] = {
#             "scores_likert": scores,
#             "percentual_uteis": useful_pct,
#             "frequencia_alucinacoes": hallucinations / len(test_samples),
#             "media_score": np.mean(scores)
#         }
#         print(f"\nJustificativas úteis (score >= 4): {useful_pct:.1f}%")
# 
#     def exp5_operational_efficiency(self):
#         """Experimento 5: Eficiência operacional (convergência vs. comunicação)."""
#         print("\n" + "=" * 60)
#         print("EXPERIMENTO 5: Eficiência Operacional da Rede")
#         print("=" * 60)
# 
#         tracker = ConvergenceTracker()
#         rounds_data = []
#         param_size_mb = 2.0  # BEHRT simplificado ~2MB
# 
#         for r in range(1, NUM_ROUNDS + 1):
#             acc = 0.70 + 0.20 * (1 - np.exp(-0.12 * r)) + np.random.normal(0, 0.01)
#             comm_cost = r * param_size_mb * NUM_CLIENTS
# 
#             converged = tracker.check(acc)
#             rounds_data.append({
#                 "rodada": r,
#                 "acuracia": acc,
#                 "custo_comunicacao_mb": comm_cost,
#                 "convergiu": converged
#             })
# 
#             if converged and r > 10:
#                 print(f"Convergência detectada na rodada {r} (T_opt)")
#                 break
# 
#         t_opt = next((d['rodada'] for d in rounds_data if d['convergiu']), NUM_ROUNDS)
# 
#         self.results["exp5"] = {
#             "rodadas": rounds_data,
#             "t_opt": t_opt,
#             "custo_total_mb": t_opt * param_size_mb * NUM_CLIENTS,
#             "acuracia_final": rounds_data[-1]['acuracia']
#         }
#         print(f"T_opt = {t_opt} rodadas | Custo total: {self.results['exp5']['custo_total_mb']:.1f} MB")
# 
#     def save_results(self, filename: str = "experiment_results.json"):
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(self.results, f, ensure_ascii=False, indent=2)
#         print(f"\nResultados salvos em {filename}")
# 
# 
# if __name__ == "__main__":
#     runner = ExperimentRunner()
# 
#     # Dados dummy para demonstração (substituir pela base FAPESP real)
#     df_dummy = pd.DataFrame({
#         'instituicao': ['HF1']*500 + ['HF2']*300 + ['HF3']*200 + ['HF4']*150 + ['HF5']*50,
#         'idade': np.concatenate([
#             np.random.normal(75, 10, 500),  # HF1: idosos
#             np.random.normal(8, 5, 300),    # HF2: crianças
#             np.random.normal(45, 15, 200),  # HF3: adultos
#             np.random.normal(60, 12, 150),  # HF4: meia-idade
#             np.random.normal(35, 10, 50)    # HF5: jovens (poucos dados)
#         ]),
#         'sintoma': np.random.choice(['febre', 'tosse', 'dispneia', 'cefaleia', 'mialgia'], 1200),
#         'exame': np.random.choice(['pcr_pos', 'pcr_neg', 'rx_normal', 'rx_opacidade'], 1200),
#         'desfecho': np.random.choice([0, 1], 1200, p=[0.7, 0.3])
#     })
# 
#     # Execução dos experimentos
#     df_proc = runner.exp1_standardization(df_dummy)
#     clients = split_by_institution(df_proc)
#     runner.exp2_equalizing_effect(clients)
#     runner.exp3_non_iid_impact(clients)
# 
#     # RAG
#     rag = ClinicalRAG()
#     patterns = [
#         {"texto": "Paciente idoso 67 anos, febre alta, saturação 92%, evolução para pneumonia",
#          "desfecho": "pneumonia", "faixa_etaria": "60-70", "categoria": "respiratorio"},
#         {"texto": "Criança 5 anos, febre moderada, tosse seca, evolução favorável",
#          "desfecho": "alta", "faixa_etaria": "0-10", "categoria": "respiratorio"}
#     ]
#     rag.build_knowledge_base(patterns)
#     test_samples = [{"febre": "alta", "tosse": "seca", "saturacao": "92%", "faixa_etaria": "60-70"} for _ in range(50)]
#     runner.exp4_rag_uncertainty(rag, test_samples)
# 
#     runner.exp5_operational_efficiency()
#     runner.save_results()
