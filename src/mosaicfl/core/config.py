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

Para restaurar os valores originais (máquinas com GPU), substitua RuntimeConfig.device por:
    device: object = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_DEFAULT_CLASS_LABELS = (
    "curado_pronto",
    "curado_internado",
    "melhora_pronto",
    "melhora_internado_breve",
    "melhora_internado_grave",
)


@dataclass(frozen=True)
class ModelConfig:
    """Arquitetura BEHRT — imutável por experimento. frozen=True impede mutação acidental em runtime."""
    vocab_size:   int            = 10_000
    embed_dim:    int            = 64
    max_seq_len:  int            = 128
    num_layers:   int            = 2
    num_heads:    int            = 4
    ff_dim:       int            = 128
    num_classes:  int            = 4
    class_labels: tuple[str, ...] = _DEFAULT_CLASS_LABELS
    dropout:      float          = 0.1

    def __post_init__(self) -> None:
        if len(self.class_labels) != self.num_classes:
            raise ValueError(
                f"FL_CLASS_LABELS tem {len(self.class_labels)} label(s) "
                f"mas FL_NUM_CLASSES={self.num_classes}. Devem ser iguais."
            )


@dataclass(frozen=True)
class FedConfig:
    """Protocolo federado e hiperparâmetros de treinamento — imutável por experimento."""
    num_rounds:            int   = 20
    fraction_fit:          float = 1.0
    fraction_evaluate:     float = 1.0
    proximal_mu:           float = 0.01
    min_fit_clients:       int   = 2
    min_evaluate_clients:  int   = 2
    min_available_clients: int   = 2
    convergence_threshold: float = 0.005
    convergence_patience:  int   = 3
    batch_size:            int   = 16
    local_epochs:          int   = 2
    lr:                    float = 0.001
    num_clients:           int   = 2
    random_seed:           int   = 42
    top_k:                 int   = 3
    max_new_tokens:        int   = 64


@dataclass
class RuntimeConfig:
    """Parâmetros que dependem do ambiente de execução — variam por máquina/deployment."""
    data_path:       Path   = field(default_factory=lambda: Path(os.getenv("FL_DATA_PATH", "data/fapesp_covid19")))
    device:          object = field(default_factory=lambda: torch.device(os.getenv("FL_DEVICE", "cpu")))
    chroma_path:     Path   = field(default_factory=lambda: Path(os.getenv("FL_CHROMA_PATH", "chroma_db")))
    embedding_model: str    = field(default_factory=lambda: os.getenv("FL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    llm_model:       str    = field(default_factory=lambda: os.getenv("FL_LLM_MODEL", "distilgpt2"))
    use_ray:         bool   = field(default_factory=lambda: os.getenv("FL_USE_RAY", "false").lower() == "true")
    db_url:          str    = field(default_factory=lambda: os.getenv("FL_DB_URL", ""))
    # "production" exige FL_DB_URL e rejeita dados sintéticos.
    # "development" (padrão) permite fallback para sintético quando FL_DB_URL não está configurado.
    env:             str    = field(default_factory=lambda: os.getenv("FL_ENV", "development").lower())


MODEL_CFG = ModelConfig(
    vocab_size   = int(os.getenv("FL_VOCAB_SIZE",   "10000")),
    embed_dim    = int(os.getenv("FL_EMBED_DIM",    "64")),
    max_seq_len  = int(os.getenv("FL_MAX_SEQ_LEN",  "128")),
    num_layers   = int(os.getenv("FL_NUM_LAYERS",   "2")),
    num_heads    = int(os.getenv("FL_NUM_HEADS",    "4")),
    ff_dim       = int(os.getenv("FL_FF_DIM",       "128")),
    num_classes  = int(os.getenv("FL_NUM_CLASSES",  "5")),
    class_labels = tuple(
        os.getenv("FL_CLASS_LABELS", ",".join(_DEFAULT_CLASS_LABELS)).split(",")
    ),
    dropout      = float(os.getenv("FL_DROPOUT",    "0.1")),
)
FED_CFG = FedConfig(
    num_rounds   = int(os.getenv("FL_NUM_ROUNDS",   "20")),
    batch_size   = int(os.getenv("FL_BATCH_SIZE",   "16")),
    local_epochs = int(os.getenv("FL_LOCAL_EPOCHS", "2")),
    lr           = float(os.getenv("FL_LR",         "0.001")),
    proximal_mu  = float(os.getenv("FL_PROXIMAL_MU","0.01")),
    num_clients  = int(os.getenv("FL_NUM_CLIENTS",  "2")),
    random_seed  = int(os.getenv("FL_RANDOM_SEED",  "42")),
)
RUNTIME_CFG = RuntimeConfig()

# Aliases de compatibilidade para testes e scripts legados — não usar em código novo
VOCAB_SIZE            = MODEL_CFG.vocab_size
EMBED_DIM             = MODEL_CFG.embed_dim
MAX_SEQ_LEN           = MODEL_CFG.max_seq_len
NUM_LAYERS            = MODEL_CFG.num_layers
NUM_HEADS             = MODEL_CFG.num_heads
FF_DIM                = MODEL_CFG.ff_dim
NUM_CLASSES           = MODEL_CFG.num_classes
DROPOUT               = MODEL_CFG.dropout

NUM_ROUNDS            = FED_CFG.num_rounds
FRACTION_FIT          = FED_CFG.fraction_fit
FRACTION_EVALUATE     = FED_CFG.fraction_evaluate
PROXIMAL_MU           = FED_CFG.proximal_mu
MIN_FIT_CLIENTS       = FED_CFG.min_fit_clients
MIN_EVALUATE_CLIENTS  = FED_CFG.min_evaluate_clients
MIN_AVAILABLE_CLIENTS = FED_CFG.min_available_clients
CONVERGENCE_THRESHOLD = FED_CFG.convergence_threshold
CONVERGENCE_PATIENCE  = FED_CFG.convergence_patience
BATCH_SIZE            = FED_CFG.batch_size
LOCAL_EPOCHS          = FED_CFG.local_epochs
LR                    = FED_CFG.lr
NUM_CLIENTS           = FED_CFG.num_clients
RANDOM_SEED           = FED_CFG.random_seed
TOP_K                 = FED_CFG.top_k
MAX_NEW_TOKENS        = FED_CFG.max_new_tokens

DATA_PATH             = RUNTIME_CFG.data_path
DEVICE                = RUNTIME_CFG.device
CHROMA_DB_PATH        = str(RUNTIME_CFG.chroma_path)
EMBEDDING_MODEL       = RUNTIME_CFG.embedding_model
LLM_MODEL             = RUNTIME_CFG.llm_model
USE_RAY               = RUNTIME_CFG.use_ray
FL_DB_URL             = RUNTIME_CFG.db_url
FL_ENV                = RUNTIME_CFG.env
