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


def _resolve_class_labels() -> tuple[str, ...]:
    """Lê FL_CLASS_LABELS (CSV) ou usa o conjunto padrão de 5 desfechos clínicos."""
    raw = os.getenv("FL_CLASS_LABELS", "").strip()
    if raw:
        return tuple(lbl.strip() for lbl in raw.split(",") if lbl.strip())
    return (
        "curado_pronto",
        "curado_internado",
        "melhora_pronto",
        "melhora_internado_breve",
        "melhora_internado_grave",
    )


_CLASS_LABELS = _resolve_class_labels()


@dataclass(frozen=True)
class ModelConfig:
    """Arquitetura BEHRT — imutável por experimento. frozen=True impede mutação acidental em runtime."""
    vocab_size:   int            = 10_000
    embed_dim:    int            = 64
    max_seq_len:  int            = 128
    num_layers:   int            = 2
    num_heads:    int            = 4
    ff_dim:       int            = 128
    num_classes:  int            = len(_CLASS_LABELS)
    class_labels: tuple[str, ...] = _CLASS_LABELS
    dropout:      float          = 0.1

    def __post_init__(self) -> None:
        if len(self.class_labels) != self.num_classes:
            raise ValueError(
                f"FL_CLASS_LABELS tem {len(self.class_labels)} label(s) "
                f"mas num_classes={self.num_classes}. Devem ser iguais."
            )


@dataclass(frozen=True)
class FedConfig:
    """Protocolo federado e hiperparâmetros de treinamento — imutável por experimento."""
    num_rounds:            int   = 120  # teto máximo; early stopping pode parar antes (ver min_rounds)
    min_rounds:            int   = 20   # warm-up: convergência só é avaliada após esta rodada
    fraction_fit:          float = 1.0
    fraction_evaluate:     float = 1.0
    proximal_mu:           float = 0.1  # aumentado de 0.01 → 0.1 (Exp 7): reduz drift não-IID (Li et al. 2020)
    min_fit_clients:       int   = 2
    min_evaluate_clients:  int   = 2
    min_available_clients: int   = 2
    convergence_threshold: float = 0.005
    convergence_patience:  int   = 3
    batch_size:            int   = 16
    local_epochs:          int   = 1   # reduzido de 2→1: menos client drift em regime non-IID severo (Li et al. 2020)
    lr:                    float = 0.001
    num_clients:           int   = 2
    random_seed:           int        = 42
    ablation_seeds:        list[int]  = field(default_factory=lambda: [42])  # deve incluir random_seed; lista única = mesmo contexto do FL
    top_k:                 int        = 3
    max_new_tokens:        int   = 64
    pooled_epochs:         int   = 120  # épocas do BEHRT centralizado — equivalente ao budget de rodadas do FL (num_rounds)
    use_fednova:           bool  = True  # Exp 9: substitui FedAvg por normalização por passos efetivos τ_i (Wang et al. 2020)
    # Critério de seleção do melhor checkpoint por rodada.
    # Valores válidos: "f1_macro" (padrão, Bloco 2+), "accuracy" (Bloco 1 — legado).
    # Futuro: migra para fl_config no banco com audit trail (justificativa + efeitos esperados).
    checkpoint_criterion:  str   = field(default_factory=lambda: os.getenv("FL_CHECKPOINT_CRITERION", "f1_macro"))
    # Método de calibração pós-treinamento (calibration_mixin._run_calibration).
    # Valores válidos:
    #   "temperature" (padrão — TemperatureScaler, Guo et al. 2017)
    #   "isotonic"    (IsotonicCalibrator OvR, Zadrozny & Elkan 2002 — histórico do projeto mostra
    #                  ECE menor que temperature scaling quando o viés de subconfiança não é
    #                  uniforme entre classes, ver docstring de mosaicfl.core.calibration)
    #   "auto"        (ajusta os dois no conjunto de calibração e persiste o que tiver menor ECE —
    #                  formaliza a comparação que experiments/training/core/fl_core/manual_loop.py
    #                  já fazia manualmente; não é combinação/ensemble dos dois, é escolha do vencedor)
    # Combinação de verdade (ensemble/ponderação simultânea dos dois calibradores) fica como ideia
    # futura — ver docs/Linha_do_Tempo_MOSAIC-FL.md (2026-07-12) e memória de projeto
    # project_calibracao_ensemble_futuro — não implementar sem lastro bibliográfico federado.
    # Nenhum dos três modos é federado sob DP ainda
    # (ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 9).
    calibration_method:    str   = field(default_factory=lambda: os.getenv("FL_CALIBRATION_METHOD", "temperature"))
    # Privacidade Diferencial (DP-FedAvg, McMahan et al. 2018)
    # dp_noise_multiplier=0.0 desabilita DP completamente (sem overhead).
    # dp_noise_multiplier=σ > 0: cada rodada adiciona N(0, (σ·S/n)²) ao modelo agregado,
    # onde S=dp_max_grad_norm (sensitivity) e n=num_clients.
    dp_noise_multiplier: float = field(default_factory=lambda: float(os.getenv("FL_DP_NOISE", "0.0")))
    dp_max_grad_norm:    float = field(default_factory=lambda: float(os.getenv("FL_DP_CLIP",  "1.0")))


@dataclass
class RuntimeConfig:
    """Parâmetros que dependem do ambiente de execução — variam por máquina/deployment."""
    data_path:       Path   = field(default_factory=lambda: Path(os.getenv("FL_DATA_PATH", "data/fapesp_covid19")))
    device:          object = field(default_factory=lambda: torch.device(os.getenv("FL_DEVICE", "cpu")))
    chroma_path:     Path   = field(default_factory=lambda: Path(os.getenv("FL_CHROMA_PATH", "chroma_db")))
    embedding_model: str    = field(default_factory=lambda: os.getenv("FL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    llm_model:       str    = field(default_factory=lambda: os.getenv("FL_LLM_MODEL", "distilgpt2"))
    llm_hf_model:   str    = field(default_factory=lambda: os.getenv("FL_LLM_HF_MODEL", "distilgpt2"))
    llm_backend:     str    = field(default_factory=lambda: os.getenv("FL_LLM_BACKEND", "huggingface"))
    use_ray:         bool   = field(default_factory=lambda: os.getenv("FL_USE_RAY", "false").lower() == "true")
    db_url:          str    = field(default_factory=lambda: os.getenv("FL_DB_URL", ""))
    # "production" exige FL_DB_URL e rejeita dados sintéticos.
    # "development" (padrão) permite fallback para sintético quando FL_DB_URL não está configurado.
    env:             str    = field(default_factory=lambda: os.getenv("FL_ENV", "development").lower())


MODEL_CFG = ModelConfig()
FED_CFG   = FedConfig()
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
MIN_ROUNDS            = FED_CFG.min_rounds
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
POOLED_EPOCHS         = FED_CFG.pooled_epochs
LR                    = FED_CFG.lr
NUM_CLIENTS           = FED_CFG.num_clients
RANDOM_SEED           = FED_CFG.random_seed
ABLATION_SEEDS        = FED_CFG.ablation_seeds
TOP_K                 = FED_CFG.top_k
MAX_NEW_TOKENS        = FED_CFG.max_new_tokens
DP_NOISE_MULTIPLIER   = FED_CFG.dp_noise_multiplier
DP_MAX_GRAD_NORM      = FED_CFG.dp_max_grad_norm

DATA_PATH             = RUNTIME_CFG.data_path
DEVICE                = RUNTIME_CFG.device
CHROMA_DB_PATH        = str(RUNTIME_CFG.chroma_path)
EMBEDDING_MODEL       = RUNTIME_CFG.embedding_model
LLM_MODEL             = RUNTIME_CFG.llm_model
LLM_BACKEND           = RUNTIME_CFG.llm_backend
USE_RAY               = RUNTIME_CFG.use_ray
FL_DB_URL             = RUNTIME_CFG.db_url
FL_ENV                = RUNTIME_CFG.env
