"""
Configurações globais para os experimentos do TCC.
Inteligência Artificial Colaborativa na Saúde: FL + RAG

Hardware alvo: Dell Inspiron 5402 — i7-1165G7 (8 threads), 16GB RAM, sem GPU dedicada.
Os valores abaixo são calibrados para rodar de forma estável nessa configuração.
Para máquinas mais potentes, aumente BATCH_SIZE, NUM_ROUNDS e NUM_CLIENTS.

Tempo estimado de execução completa (5 experimentos): 15–25 minutos

Tabela de parâmetros ajustados em relação aos valores originais:
┌──────────────────────────┬─────────┬─────────┬────────────────────────────────────────────────┐
│ Parâmetro                │  Antes  │ Depois  │ Motivo                                         │
├──────────────────────────┼─────────┼─────────┼────────────────────────────────────────────────┤
│ OMP/MKL_NUM_THREADS      │ —       │ 4       │ Libera 4 threads para o SO, evita travamento   │
│ TOKENIZERS_PARALLELISM   │ —       │ false   │ Elimina conflito de threads do HuggingFace     │
│ DEVICE                   │ cuda*   │ cpu     │ Intel Iris Xe não tem suporte CUDA             │
│ BATCH_SIZE               │ 32      │ 16      │ Reduz uso de RAM por cliente                   │
│ LOCAL_EPOCHS             │ 3       │ 2       │ Menos iterações por rodada federada            │
│ NUM_ROUNDS               │ 50      │ 20      │ Principal causa do barulho do cooler           │
│ MAX_NEW_TOKENS           │ 100     │ 64      │ Geração de texto mais rápida no RAG            │
└──────────────────────────┴─────────┴─────────┴────────────────────────────────────────────────┘
  * cuda if available — tentativa inútil sem GPU dedicada

Para restaurar os valores originais (máquinas com GPU), reverta os parâmetros
marcados acima e substitua DEVICE por:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
import os
import torch

# Limita threads de álgebra linear para evitar sobrecarga da CPU
# O i7-1165G7 tem 4 núcleos / 8 threads — reservamos 4 para o SO
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["MKL_NUM_THREADS"]        = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # evita warning do HuggingFace

# Dados
DATA_PATH   = "data/fapesp_covid19"
RANDOM_SEED = 42

# Modelo BEHRT Simplificado
VOCAB_SIZE  = 10000
EMBED_DIM   = 64
MAX_SEQ_LEN = 128
NUM_LAYERS  = 2
NUM_HEADS   = 4
FF_DIM      = 128
NUM_CLASSES = 2
DROPOUT     = 0.1

# Treinamento
# BATCH_SIZE 16 usa ~2GB RAM por cliente — seguro com 16GB
BATCH_SIZE   = 16   # era 32 → reduzido para poupar RAM
LOCAL_EPOCHS = 2    # era 3  → reduzido para rodar mais rápido
LR           = 0.001
DEVICE       = torch.device("cpu")  # i7-1165G7 não tem GPU compatível com CUDA

# Federado (Flower + FedProx)
# NUM_ROUNDS 20 já é suficiente para demonstrar convergência no TCC
NUM_ROUNDS            = 20   # era 50 → principal responsável pelo barulho do cooler
FRACTION_FIT          = 1.0
FRACTION_EVALUATE     = 1.0
PROXIMAL_MU           = 0.01
MIN_FIT_CLIENTS       = 3
MIN_EVALUATE_CLIENTS  = 3
MIN_AVAILABLE_CLIENTS = 3
CONVERGENCE_THRESHOLD = 0.005
CONVERGENCE_PATIENCE  = 3

# RAG
CHROMA_DB_PATH  = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "distilgpt2"
TOP_K           = 3
MAX_NEW_TOKENS  = 64   # era 100 → reduzido para geração mais rápida

# Experimentos
NUM_CLIENTS = 5

USE_RAY = False