"""
Configurações globais para os experimentos do TCC.
Inteligência Artificial Colaborativa na Saúde: FL + RAG
"""
import torch

# Dados
DATA_PATH = "data/fapesp_covid19"
RANDOM_SEED = 42

# Modelo BEHRT Simplificado
VOCAB_SIZE = 10000
EMBED_DIM = 64
MAX_SEQ_LEN = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 128
NUM_CLASSES = 2
DROPOUT = 0.1

# Treinamento
BATCH_SIZE = 32
LOCAL_EPOCHS = 3
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Federado (Flower + FedProx)
NUM_ROUNDS = 50
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
PROXIMAL_MU = 0.01
MIN_FIT_CLIENTS = 3
MIN_EVALUATE_CLIENTS = 3
MIN_AVAILABLE_CLIENTS = 3
CONVERGENCE_THRESHOLD = 0.005
CONVERGENCE_PATIENCE = 3

# RAG
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "distilgpt2"
TOP_K = 3
MAX_NEW_TOKENS = 100

# Experimentos
NUM_CLIENTS = 5
