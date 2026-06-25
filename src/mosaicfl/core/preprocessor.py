"""
Pré-processamento e padronização da base FAPESP COVID-19 Data Sharing/BR — VERSÃO CORRIGIDA.

Mudanças principais:
  1. clean_text preserva pontos (.) e hífens (-) essenciais para códigos ICD e valores decimais.
  2. Validação de colunas antes de acessar (evita KeyError silencioso).
  3. Estratégia de missing configurável por coluna (nem sempre impute é adequado).
  4. split_by_institution agora embaralha e estratifica para balancear desfechos.
  5. Logging de rejeição por coluna (não apenas global).
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EHRPreprocessor:
    def __init__(self):
        self.vocab_map: Dict[str, int] = {}
        self.unit_conversions = {
            'peso': {'lb': 0.453592, 'lbs': 0.453592, 'kg': 1.0, 'g': 0.001},
            'idade': {'meses': 1/12, 'anos': 1.0, 'dias': 1/365.25},
            'temperatura': {'f': lambda x: (x - 32) * 5/9, 'c': lambda x: x}
        }
        self.transform_log: List[Dict] = []
        self.rejected_count = 0
        self.total_count = 0

    def _log_transform(self, step: str, detail: str, count: int = 0) -> None:
        entry = {"step": step, "detail": detail, "count": count}
        self.transform_log.append(entry)
        logger.info(f"[{step}] {detail} (n={count})")

    def normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'idade_unidade' in df.columns and 'idade' in df.columns:
            mask_meses = df['idade_unidade'].str.lower().isin(['meses', 'm'])
            mask_dias = df['idade_unidade'].str.lower().isin(['dias', 'd'])
            #df.loc[mask_meses, 'idade'] = df.loc[mask_meses, 'idade'] / 12.0
            df['idade'] = df['idade'].astype(float)
            df.loc[mask_meses, 'idade'] = df.loc[mask_meses, 'idade'] / 12.0
            df.loc[mask_dias, 'idade'] = df.loc[mask_dias, 'idade'] / 365.25
            df.loc[:, 'idade_unidade'] = 'anos'
            self._log_transform("unidade", "Idade normalizada para anos", int(mask_meses.sum() + mask_dias.sum()))

        if 'peso_unidade' in df.columns and 'peso' in df.columns:
            # Garante dtype float antes de atribuir resultado de multiplicação.
            # Pandas 2.x+ recusa atribuição de float em coluna int64 (LossySetitemError).
            df['peso'] = df['peso'].astype(float)
            for unit, factor in self.unit_conversions['peso'].items():
                if isinstance(factor, float):
                    mask = df['peso_unidade'].str.lower() == unit
                    df.loc[mask, 'peso'] = df.loc[mask, 'peso'] * factor
            df.loc[:, 'peso_unidade'] = 'kg'
            self._log_transform("unidade", "Peso normalizado para kg")
        return df

    def build_vocabulary(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, int]:
        vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3}
        idx = 4
        for col in text_cols:
            if col not in df.columns:
                logger.warning(f"Coluna '{col}' não encontrada — pulando vocabulário.")
                continue
            unique_vals = df[col].dropna().astype(str).unique()
            for val in unique_vals:
                if val not in vocab:
                    vocab[val] = idx
                    idx += 1
        self.vocab_map = vocab
        self._log_transform("vocab", f"Vocabulário construído: {len(vocab)} tokens (inclui <CLS>)")
        return vocab

    def encode_sequences(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in text_cols:
            if col in df.columns:
                df[col + '_encoded'] = df[col].astype(str).map(self.vocab_map).fillna(1).astype(int)
        return df

    # def handle_missing(self, df: pd.DataFrame, strategy: str = "impute",
    #                    numeric_strategy: str = "median",
    #                    categorical_strategy: str = "<UNK>") -> pd.DataFrame:
    #     before = len(df)
    #     if strategy == "drop":
    #         df = df.dropna(subset=df.columns[df.isnull().any()])
    #         self.rejected_count += before - len(df)
    #         self._log_transform("missing", "Registros removidos por valores ausentes", before - len(df))
    #     elif strategy == "impute":
    #         numeric_cols = df.select_dtypes(include=[np.number]).columns
    #         cat_cols = df.select_dtypes(include=['object']).columns

    #         for col in numeric_cols:
    #             if df[col].isnull().any():
    #                 if numeric_strategy == "median":
    #                     fill_val = df[col].median()
    #                 elif numeric_strategy == "mean":
    #                     fill_val = df[col].mean()
    #                 else:
    #                     fill_val = 0
    #                 df[col] = df[col].fillna(fill_val)
    #                 self._log_transform("missing", f"Coluna '{col}': preenchido com {numeric_strategy}={fill_val:.2f}")

    #         for col in cat_cols:
    #             if df[col].isnull().any():
    #                 df[col] = df[col].fillna(categorical_strategy)
    #                 self._log_transform("missing", f"Coluna '{col}': preenchido com '{categorical_strategy}'")
    #     return df

    def handle_missing(self, df: pd.DataFrame, strategy: str = "impute") -> pd.DataFrame:
        before = len(df)
        if strategy == "drop":
            df = df.dropna(subset=df.columns[df.isnull().any()])
            self.rejected_count += before - len(df)
            self._log_transform("missing", "Registros removidos por valores ausentes", before - len(df))
        elif strategy == "impute":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include=['object', 'str']).columns:
                df[col] = df[col].fillna("<UNK>")
            self._log_transform("missing", "Valores ausentes imputados (mediana/UNK)")
        return df


    # def clean_text(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    #     """
    #     Limpa texto preservando pontuação médica essencial:
    #       - Pontos (.) em códigos ICD (J18.1) e valores decimais (98.6)
    #       - Hífens (-) em nomes compostos e ranges
    #     Remove apenas caracteres especiais que não têm valor semântico clínico.
    #     """
    #     df = df.copy()
    #     for col in cols:
    #         if col not in df.columns:
    #             logger.warning(f"Coluna '{col}' não encontrada em clean_text — pulando.")
    #             continue
    #         # Lowercase e trim
    #         df[col] = df[col].astype(str).str.lower().str.strip()
    #         # Preserva: letras, números, espaços, pontos (ICD, decimais), hífens
    #         # Remove: outros caracteres especiais (!@#$% etc.)
    #         df[col] = df[col].str.replace(r'[^\w\s\.\-]', '', regex=True)
    #         # Remove espaços múltiplos
    #         df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    #     self._log_transform("clean", "Texto limpo: lowercase, trim, preserva . e - (ICD/decimais)")
    #     return df

    def clean_text(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].str.replace(r'[^\w\s\-]', '', regex=True)
        self._log_transform("clean", "Texto limpo: lowercase, trim, remoção de pontuação")
        return df




    def process(self, df: pd.DataFrame, text_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        self.total_count = len(df)
        text_cols = text_cols or ['sintoma', 'exame', 'diagnostico']
        df = self.clean_text(df, text_cols)
        df = self.normalize_units(df)
        df = self.handle_missing(df, strategy="impute")
        self.build_vocabulary(df, text_cols)
        df = self.encode_sequences(df, text_cols)
        summary = {
            "total_amostras": self.total_count,
            "amostras_rejeitadas": self.rejected_count,
            "percentual_rejeitado": round(self.rejected_count / self.total_count * 100, 2) if self.total_count else 0,
            "tamanho_vocabulario": len(self.vocab_map),
            "transformacoes": self.transform_log
        }
        logger.info(f"Pré-processamento concluído. Resumo: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        return df, summary


# def split_by_institution(
#     df: pd.DataFrame,
#     institution_col: str = 'instituicao',
#     num_clients: int = 5,
#     stratify_col: Optional[str] = None,
#     random_state: int = 42,
# ) -> Dict[int, pd.DataFrame]:
#     """
#     Divide dados por instituição, com opção de estratificação por desfecho.

#     Args:
#         stratify_col: se fornecido (ex: 'desfecho'), embaralha e estratifica
#                       para garantir distribuição balanceada entre clientes.
#     """
#     clients = {}
#     institutions = df[institution_col].unique()

#     if len(institutions) < num_clients:
#         logger.warning(f"Apenas {len(institutions)} instituições encontradas — "
#                        f"ajustando num_clients de {num_clients} para {len(institutions)}")
#         num_clients = len(institutions)

#     for i, inst in enumerate(institutions[:num_clients]):
#         subset = df[df[institution_col] == inst].copy()

#         if stratify_col and stratify_col in subset.columns:
#             # Embaralha estratificado para evitar que um cliente fique com
#             # apenas um desfecho (extremo non-IID não-intencional)
#             subset = subset.groupby(stratify_col, group_keys=False).apply(
#                 lambda x: x.sample(frac=1, random_state=random_state)
#             ).reset_index(drop=True)
#         else:
#             subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)

#         clients[i] = subset
#         desfecho_dist = subset[stratify_col].value_counts().to_dict() if stratify_col else "N/A"
#         logger.info(f"Cliente {i} ({inst}): {len(clients[i])} registros | Distribuição: {desfecho_dist}")

#     return clients

# def split_by_institution(df: pd.DataFrame, institution_col: str = 'instituicao', num_clients: int = 5) -> Dict[int, pd.DataFrame]:
#     clients = {}
#     institutions = df[institution_col].unique()
#     for i, inst in enumerate(institutions[:num_clients]):
#         clients[i] = df[df[institution_col] == inst].copy()
#         logger.info(f"Cliente {i} ({inst}): {len(clients[i])} registros")
#     return clients

def split_by_institution(
    df: pd.DataFrame,
    institution_col: str = 'instituicao',
    num_clients: int = 5,
    stratify_col: str = None,
    random_state: int = None,
) -> Dict[int, pd.DataFrame]:
    """
    Divide o DataFrame por instituição, criando um cliente FL por hospital.

    Args:
        df:               DataFrame processado.
        institution_col:  Coluna com o identificador da instituição.
        num_clients:      Número máximo de clientes (hospitais).
        stratify_col:     Se informado, loga a distribuição dessa coluna por cliente
                          (útil para verificar balanceamento de desfechos entre hospitais).
        random_state:     Semente para embaralhamento antes da divisão (reprodutibilidade).

    Returns:
        Dict {client_id: DataFrame} com um subset por instituição.
    """
    clients = {}
    institutions = df[institution_col].unique()

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        institutions = rng.permutation(institutions)

    for i, inst in enumerate(institutions[:num_clients]):
        subset = df[df[institution_col] == inst].copy()
        clients[i] = subset

        if stratify_col and stratify_col in subset.columns:
            dist = subset[stratify_col].value_counts().to_dict()
            logger.info(f"Cliente {i} ({inst}): {len(subset)} registros | {stratify_col}: {dist}")
        else:
            logger.info(f"Cliente {i} ({inst}): {len(subset)} registros")

    return clients


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES DE SÉRIE TEMPORAL PARA BEHRT
# ─────────────────────────────────────────────────────────────────────────────

from .config import MODEL_CFG


# outcome_class vem de classify_outcome() em integration/fapesp/transforms.py.
# O mapeamento para 5 classes é feito por _map_outcome() combinando outcome_class,
# duration_days e attendance_type. Classes 2, 3, 4 são excluídas na query SQL.
def _map_outcome(outcome_class: int, duration_days: float, attendance_type: str) -> int:
    """
    Converte (outcome_class, duration_days, attendance_type) em classe de prognóstico.

    5 classes que cruzam desfecho clínico, tipo de atendimento e duração:

      0 = curado_pronto           — outcome 0, não-internado (ambulatorial/pronto/externo)
      1 = curado_internado        — outcome 0, internado (curso grave com recuperação completa)
      2 = melhora_pronto          — outcome 1, não-internado (COVID moderado, melhora sem internação)
      3 = melhora_internado_breve — outcome 1, internado, ≤ 10 dias
      4 = melhora_internado_grave — outcome 1, internado, > 10 dias

    O tipo de atendimento define a trajetória clínica: ambulatorial/pronto = acesso pontual;
    internado = internação contínua com acompanhamento diário de exames.

    Retorna -1 para outcome_class não mapeado (dado censurado ou excluído da análise).
    """
    internado = str(attendance_type).strip() == "Internado"

    if outcome_class == 0:
        return 1 if internado else 0
    if outcome_class == 1:
        if not internado:
            return 2
        return 3 if duration_days <= 10 else 4
    return -1  # 4=censored, 5=uti, 6=obito — excluídos ou ausentes nos dados FAPESP


class TokenMode:
    """Modos de composição de token para o pipeline de treino.

    FULL            — analito + classificação: LEUCOCITOS_HIGH  (padrão)
    ANALYTE_ONLY    — apenas o analito:        LEUCOCITOS
    CLASS_ONLY      — apenas a classificação:  HIGH

    Permite experimentar diferentes hipóteses de treino sem recarregar dados:
    - FULL:         o nível clínico importa junto com o analito
    - ANALYTE_ONLY: basta saber que o exame foi solicitado (perfil de investigação)
    - CLASS_ONLY:   padrão de anormalidade independente do analito
    """
    FULL         = "FULL"
    ANALYTE_ONLY = "ANALYTE_ONLY"
    CLASS_ONLY   = "CLASS_ONLY"


def _make_token(analyte: str, classification: str, mode: str = TokenMode.FULL) -> str:
    """Gera token a partir do analito canônico e sua classificação clínica.

    analyte        — nome canônico em maiúsculas (ex: LEUCOCITOS)
    classification — HIGH | NORMAL | LOW | NO_REF (gravado em exam_records)
    mode           — TokenMode.FULL | ANALYTE_ONLY | CLASS_ONLY
    """
    if mode == TokenMode.ANALYTE_ONLY:
        return analyte
    if mode == TokenMode.CLASS_ONLY:
        return classification
    # FULL: sem referência disponível retorna só o analito
    if classification == "NO_REF":
        return analyte
    return f"{analyte}_{classification}"


_SQL_ATENDIMENTOS = """
WITH _ranked AS (
    SELECT
        a.patient_id,
        a.attendance_id,
        a.hospital_id,
        a.attendance_type,
        p.sex,
        p.birth_year,
        co.outcome_class,
        (co.outcome_at - a.attended_at)     AS duration_days,
        e.analyte,
        e.classification,
        GREATEST(0, e.date - a.attended_at) AS dia_relativo,
        ROW_NUMBER() OVER (
            PARTITION BY a.attendance_id
            ORDER BY GREATEST(0, e.date - a.attended_at), e.analyte
        ) AS _rn
    FROM  clinical.attendances         a
    JOIN  clinical.patients            p  ON p.patient_id    = a.patient_id
    JOIN  metrics.clinical_outcomes    co ON co.attendance_id = a.attendance_id
    JOIN  metrics.exam_records         e  ON e.attendance_id  = a.attendance_id
    WHERE co.outcome_class NOT IN (2, 3, 4)
      AND a.hospital_id    IN ('HSL', 'BPSP')
      AND e.analyte        IS NOT NULL
      AND e.classification IS NOT NULL
      AND (co.outcome_at - a.attended_at) >= 0
)
SELECT patient_id, attendance_id, hospital_id, attendance_type, sex, birth_year,
       outcome_class, duration_days, analyte, classification, dia_relativo
FROM   _ranked
WHERE  _rn <= :max_seq_len
ORDER  BY patient_id, attendance_id, dia_relativo, analyte
"""


class SequencePipelineInicial:
    """
    Abordagem inicial para construção de sequências clínicas — preservada como referência.

    Contexto histórico
    ------------------
    Primeira tentativa de estruturar os dados FAPESP como entrada para o BEHRT.
    O label binário (desfecho 0=alta / 1=outro) foi a hipótese natural de partida,
    mas revelou-se inviável pelos seguintes motivos:

    1. A base FAPESP não registra óbito como outcome_class distinto — os desfechos
       disponíveis são tipos de saída hospitalar (alta, administrativa, transferência,
       evasão), sem discriminar morte do paciente.
    2. A abordagem não aproveitava a dimensão temporal dos exames: a sequência era
       a concatenação plana dos tokens disponíveis para o paciente, sem âncora de
       tempo relativo à admissão (dia_relativo).
    3. Sem filtro por tipo de atendimento: ambulatorial e internados eram misturados,
       criando distribuições incomparáveis entre hospitais.

    Por que foi substituída
    -----------------------
    Essas limitações levaram à abordagem de faixas de tempo de internação
    (ver SequencePipeline), onde o label reflete complexidade clínica real e a
    sequência é ancorada temporalmente na data de admissão.

    Interface
    ---------
    Recebe um DataFrame já pré-processado pelo EHRPreprocessor, com:
    - coluna ``exame_encoded`` (int): token ID do exame, gerado por build_vocabulary()
    - coluna ``desfecho`` (int): label binário 0/1
    - coluna identificadora de paciente (padrão: ``patient_id``)

    Uso::

        preprocessor = EHRPreprocessor()
        df, _ = preprocessor.process(raw_df, text_cols=["exame"])
        pipeline = SequencePipelineInicial()
        sequences, labels = pipeline.build(df)
        # sequences: torch.LongTensor (n_pacientes, max_seq_len)
        # labels:    torch.LongTensor (n_pacientes,)  — 0 ou 1
    """

    def __init__(
        self,
        patient_col: str = "patient_id",
        max_seq_len: int = MODEL_CFG.max_seq_len,
    ):
        self.patient_col = patient_col
        self.max_seq_len = max_seq_len

    def build(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constrói tensores de sequência e label a partir de um DataFrame pré-processado."""
        if self.patient_col not in df.columns:
            logger.warning(
                "Coluna '%s' ausente — tratando o DataFrame inteiro como um único paciente.",
                self.patient_col,
            )
            tokens = df["exame_encoded"].dropna().astype(int).tolist()
            label = int(df["desfecho"].iloc[0]) if "desfecho" in df.columns else 0
            return (
                torch.tensor([self._pad(tokens)], dtype=torch.long),
                torch.tensor([label], dtype=torch.long),
            )

        sequences: List[List[int]] = []
        labels: List[int] = []
        for _, group in df.groupby(self.patient_col):
            tokens = group["exame_encoded"].dropna().astype(int).tolist()
            sequences.append(self._pad(tokens))
            labels.append(int(group["desfecho"].iloc[0]) if "desfecho" in group.columns else 0)

        return (
            torch.tensor(sequences, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def _pad(self, tokens: List[int]) -> List[int]:
        tokens = tokens[: self.max_seq_len]
        return tokens + [0] * (self.max_seq_len - len(tokens))


class SequencePipeline:
    """
    Pipeline de série temporal clínica para BEHRT — Trajetória Clínica por Tipo de Atendimento.

    Motivação
    ---------
    A base FAPESP COVID-19 não contém códigos diagnósticos (ICD-10) por decisão de
    privacidade (LGPD). Na ausência de diagnóstico como label, a combinação de
    tipo de atendimento + desfecho clínico + duração é usada como proxy de severidade:
    a sequência temporal dos exames captura a trajetória de evolução (ou agravamento)
    dos marcadores laboratoriais ao longo do atendimento.

    O aprendizado federado (FL) viabiliza compartilhar esse padrão entre hospitais
    sem expor dados individuais, permitindo que instituições menores se beneficiem
    de casos raros observados por parceiros da rede.

    População-alvo
    --------------
    - Hospitais: HSL e BPSP — únicos com vínculo ``attendance_id`` nos exames
      (HEI: 0 % de vinculação; HFL/HCSP: sem exames vinculados a atendimentos).
    - Todos os tipos de atendimento: Internado, Ambulatorial, Pronto Socorro / Pronto
      Atendimento, Externo.
    - Exclusões: ``outcome_class IN (2, 3, 4)``
        • 2 = alta administrativa (saída burocrática, sem relação com evolução clínica)
        • 3 = transferência (desfecho clínico desconhecido)
        • 4 = em atendimento (dado censurado — desfecho final desconhecido)
    - Duração mínima: ≥ 0 dias (inclui atendimentos de mesmo dia, ex.: pronto socorro)

    Label — 5 classes que cruzam desfecho × tipo de atendimento × duração
    -----------------------------------------------------------------------
    +--------+---------------------------+-----------------------------------------------+
    | Classe | Nome                      | Critério                                      |
    +--------+---------------------------+-----------------------------------------------+
    |   0    | curado_pronto             | outcome 0, não-internado (pronto/ambul/ext)   |
    |   1    | curado_internado          | outcome 0, internado (qualquer duração)        |
    |   2    | melhora_pronto            | outcome 1, não-internado                       |
    |   3    | melhora_internado_breve   | outcome 1, internado, ≤ 10 dias               |
    |   4    | melhora_internado_grave   | outcome 1, internado, > 10 dias               |
    +--------+---------------------------+-----------------------------------------------+

    O tipo de atendimento define a trajetória clínica e é parte do label (não feature de
    entrada): o modelo aprende a inferir severidade do caso a partir dos padrões de exame.

    Classes excluídas na query: 2 (alta administrativa), 3 (transferência), 4 (censurado).

    Atenção: o modelo BEHRT usa ``MODEL_CFG.num_classes`` para dimensionar o
    classificador. Para usar este pipeline, configure ``num_classes = 5`` antes
    de instanciar ``SimplifiedBEHRT``.

    Sequência temporal
    ------------------
    - Âncora: ``attended_at`` (data de admissão do atendimento)
    - ``dia_relativo = exam_date − attended_at`` (dias desde admissão; ≥ 0)
    - Token: ``{analyte}_{bucket}`` onde bucket ∈ {baixo, normal, alto} conforme
      os campos ``ref_low`` / ``ref_high`` do próprio exame. Quando não há referência
      disponível (ref_low = ref_high = 0), usa apenas o nome do analito.
    - Ordem: cronológica (dia_relativo ASC), desempate por nome do analito.
    - Vocabulário especial: PAD=0, UNK=1, CLS=2.
      O token ``<CLS>`` é inserido pelo modelo (``SimplifiedBEHRT``), não pelo pipeline.
    - Padding / truncamento: todas as sequências ajustadas para ``max_seq_len`` com PAD (id=0).

    Fontes de dados (PostgreSQL)
    ----------------------------
    - ``clinical.attendances``       — admissões e tipo de atendimento
    - ``metrics.clinical_outcomes``  — desfecho e data de saída
    - ``metrics.exam_records``       — analitos, valores e datas dos exames

    Uso::

        pipeline = SequencePipeline(
            connection_string="postgresql://user:pass@localhost:5432/mosaicfl"
        )
        sequences, labels, vocab = pipeline.build()
        # sequences: torch.LongTensor (n_pacientes, max_seq_len)
        # labels:    torch.LongTensor (n_pacientes,) — classes 0..4
        # vocab:     Dict[str, int] para reutilizar em inferência e no RAG
    """

    _SPECIAL: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}

    def __init__(
        self,
        connection_string: str,
        max_seq_len: int = MODEL_CFG.max_seq_len,
        max_vocab_size: int = MODEL_CFG.vocab_size,
        hospital_id: Optional[str] = None,
        token_mode: str = TokenMode.FULL,
    ):
        """
        Args:
            hospital_id: Se especificado, filtra apenas esse hospital.
                         Modo produção: cada cliente FL passa seu próprio hospital_id.
                         Modo simulação: deixar None e usar build_per_hospital().
            token_mode:  TokenMode.FULL (padrão) | ANALYTE_ONLY | CLASS_ONLY.
                         Controla como analyte e classification são combinados no token.
        """
        self.connection_string = connection_string
        self.max_seq_len = max_seq_len
        self.max_vocab_size = max_vocab_size
        self.hospital_id = hospital_id
        self.token_mode = token_mode

    def build(
        self,
        vocab: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Constrói tensores de sequência, labels e vocabulário a partir da base PostgreSQL.

        Se ``hospital_id`` foi definido no construtor, filtra apenas esse hospital.

        Args:
            vocab: Vocabulário pré-compartilhado (standard_vocab.json).
                   Se fornecido, _build_vocab() é ignorado — todos os clientes FL
                   usam o mesmo espaço de tokens, tornando a agregação federada válida.
                   Se None, constrói o vocab a partir dos dados locais (modo simulação).

        Returns:
            sequences: torch.LongTensor shape (n_pacientes, max_seq_len)
            labels:    torch.LongTensor shape (n_pacientes,) — classes 0..4
            vocab:     Dict {token: id} para reutilização em inferência e RAG
        """
        import time

        def _log(msg: str) -> None:
            logger.info(msg)
            print(msg, flush=True)

        t0 = time.time()
        df = self._load_dataframe()

        if self.hospital_id:
            df = df[df["hospital_id"] == self.hospital_id].copy()
            if df.empty:
                raise RuntimeError(
                    f"Nenhum registro para hospital_id='{self.hospital_id}'. "
                    "Verifique se o hospital existe na base."
                )
            _log(f"[pipeline] filtrado para hospital_id='{self.hospital_id}' — {len(df):,} linhas")

        n_atendimentos = df.groupby(["patient_id", "attendance_id"]).ngroups
        _log(f"[pipeline] {n_atendimentos:,} atendimentos únicos encontrados")

        if vocab is not None:
            _log(f"[pipeline] usando vocabulário pré-compartilhado — {len(vocab):,} tokens")
        else:
            _log("[pipeline] construindo vocabulário local (simulação)...")
            t_v = time.time()
            vocab = self._build_vocab(df)
            _log(f"[pipeline] vocabulário pronto em {time.time() - t_v:.1f}s — {len(vocab):,} tokens")

        _log(f"[pipeline] construindo tensores para {n_atendimentos:,} sequências...")
        t_t = time.time()
        sequences, labels, _, demographics = self._build_tensors(df, vocab)
        _log(f"[pipeline] tensores prontos em {time.time() - t_t:.1f}s")

        label_dist = {i: int((labels == i).sum()) for i in range(MODEL_CFG.num_classes)}
        n_with_sex = int((demographics[:, 1] != 0).sum()) + int((demographics[:, 1] == 0).sum())
        n_male = int((demographics[:, 1] == 1.0).sum())
        _log(
            f"[pipeline] concluído em {time.time() - t0:.1f}s total — "
            f"{len(sequences):,} pacientes | dist={label_dist} | "
            f"demográficos: age_mean={demographics[:, 0].mean():.2f} "
            f"sex_M={n_male}/{n_with_sex}"
        )
        return sequences, labels, vocab, demographics

    def build_per_hospital(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]]:
        """
        Executa uma única query e divide os resultados por hospital.

        Uso exclusivo em modo de simulação FL (todos os dados em um único banco).
        Em produção, cada cliente FL instancia ``SequencePipeline(hospital_id=...)``
        e chama ``build()`` separadamente — o isolamento é garantido pelo banco local.

        Returns:
            Dict {hospital_id: (sequences, labels, vocab)}
            O vocabulário é global (construído com todos os hospitais) para garantir
            consistência entre clientes durante a simulação.
        """
        import time

        def _log(msg: str) -> None:
            logger.info(msg)
            print(msg, flush=True)

        t0 = time.time()
        df = self._load_dataframe()

        hospitals = sorted(df["hospital_id"].dropna().unique().tolist())
        _log(f"[pipeline/per_hospital] hospitais encontrados: {hospitals}")

        _log("[pipeline/per_hospital] construindo vocabulário global...")
        t_v = time.time()
        vocab = self._build_vocab(df)
        _log(f"[pipeline/per_hospital] vocabulário: {len(vocab):,} tokens em {time.time() - t_v:.1f}s")

        result: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]] = {}
        for hosp in hospitals:
            df_h = df[df["hospital_id"] == hosp].copy()
            n = df_h.groupby(["patient_id", "attendance_id"]).ngroups
            _log(f"[pipeline/per_hospital] {hosp}: construindo tensores ({n:,} atendimentos)...")
            t_h = time.time()
            seqs, lbls, _, demo = self._build_tensors(df_h, vocab)
            dist = {i: int((lbls == i).sum()) for i in range(MODEL_CFG.num_classes)}
            _log(
                f"[pipeline/per_hospital] {hosp}: {len(seqs):,} sequências "
                f"em {time.time() - t_h:.1f}s | dist={dist} | "
                f"age_mean={demo[:, 0].mean():.2f} sex_M={int((demo[:, 1] == 1.0).sum())}"
            )
            result[hosp] = (seqs, lbls, vocab, demo)

        _log(f"[pipeline/per_hospital] concluído em {time.time() - t0:.1f}s total")
        return result

    def _load_dataframe(self) -> pd.DataFrame:
        """Conecta ao banco, executa a query principal e retorna o DataFrame bruto."""
        import time
        from sqlalchemy import create_engine, text

        def _log(msg: str) -> None:
            logger.info(msg)
            print(msg, flush=True)

        _log(f"[pipeline] conectando ao banco: {self.connection_string[:50]}...")
        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            _log("[pipeline] conexão OK")
        except Exception as e:
            raise RuntimeError(f"[pipeline] falha na conexão com o banco: {e}") from e

        _log(f"[pipeline] executando query (max_seq_len={self.max_seq_len} — pode levar alguns minutos)...")
        t_q = time.time()
        with engine.connect() as conn:
            df = pd.read_sql(text(_SQL_ATENDIMENTOS), conn, params={"max_seq_len": self.max_seq_len})
        _log(f"[pipeline] query concluída em {time.time() - t_q:.1f}s — {len(df):,} linhas")

        if df.empty:
            raise RuntimeError(
                "Nenhum registro retornado. Verifique connection_string e o schema do banco."
            )
        return df

    def _build_vocab(self, df: pd.DataFrame) -> Dict[str, int]:
        has_class = "classification" in df.columns
        if self.token_mode == TokenMode.ANALYTE_ONLY:
            token_series = df["analyte"].astype(str)
        elif self.token_mode == TokenMode.CLASS_ONLY:
            token_series = df["classification"].astype(str) if has_class else df["analyte"].astype(str)
        else:  # FULL — vectorized: avoids apply(axis=1)
            if has_class:
                token_series = pd.Series(
                    np.where(
                        df["classification"] == "NO_REF",
                        df["analyte"].astype(str),
                        df["analyte"].astype(str) + "_" + df["classification"].astype(str),
                    )
                )
            else:
                token_series = df["analyte"].astype(str)

        available = self.max_vocab_size - len(self._SPECIAL)
        top_tokens = token_series.value_counts().index[:available].tolist()

        vocab = dict(self._SPECIAL)
        for i, tok in enumerate(top_tokens):
            vocab[tok] = len(self._SPECIAL) + i

        logger.info("vocab_built mode=%s tokens=%d total=%d", self.token_mode, len(top_tokens), len(vocab))
        return vocab

    # Ano de referência do dataset FAPESP COVID-19 (internações 2020-2021).
    # Usado para calcular age_at_admission = _FAPESP_REF_YEAR - birth_year.
    _FAPESP_REF_YEAR: int = 2021

    def _build_tensors(
        self, df: pd.DataFrame, vocab: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        Retorna (sequences, labels, hospital_ids, demographics).

        Token building e demográficos são computados vetorizadamente no DataFrame
        inteiro antes do groupby — elimina iterrows() no inner loop.

        demographics: FloatTensor shape (N, 2) — [age_norm, sex_binary] por paciente.
            age_norm   = (REF_YEAR − birth_year) / 100.0, clamped [0.0, 1.0]
                         0.5 quando birth_year ausente
            sex_binary = 1.0 para 'M', 0.0 para 'F' ou desconhecido
        """
        unk_id = self._SPECIAL["<UNK>"]

        # ── Vectorized token building ─────────────────────────────────────────
        has_class = "classification" in df.columns
        if self.token_mode == TokenMode.FULL and has_class:
            raw_tokens = np.where(
                df["classification"] == "NO_REF",
                df["analyte"].astype(str),
                df["analyte"].astype(str) + "_" + df["classification"].astype(str),
            )
        elif self.token_mode == TokenMode.ANALYTE_ONLY:
            raw_tokens = df["analyte"].astype(str).values
        else:
            raw_tokens = df["classification"].astype(str).values if has_class else df["analyte"].astype(str).values

        token_ids = pd.Series(raw_tokens, dtype=str).map(vocab).fillna(unk_id).astype(int).values
        df = df.assign(_token_id=token_ids)

        # ── Vectorized demographics ───────────────────────────────────────────
        if "sex" in df.columns:
            sex_bin_col = (df["sex"].astype(str).str.strip().str.upper() == "M").astype(float).values
        else:
            sex_bin_col = np.full(len(df), 0.0)

        if "birth_year" in df.columns:
            by_col = pd.to_numeric(df["birth_year"], errors="coerce")
            age_norm_col = ((self._FAPESP_REF_YEAR - by_col) / 100.0).clip(0.0, 1.0).fillna(0.5).values
        else:
            age_norm_col = np.full(len(df), 0.5)

        df = df.assign(_sex_bin=sex_bin_col, _age_norm=age_norm_col)

        # ── Group and aggregate ───────────────────────────────────────────────
        # SQL já retorna ORDER BY patient_id, attendance_id, dia_relativo, analyte
        # — dentro de cada grupo os tokens já estão na ordem correta.
        sequences: List[List[int]] = []
        labels: List[int] = []
        hospital_ids: List[str] = []
        demographics: List[List[float]] = []

        grouped = df.groupby(["patient_id", "attendance_id"], sort=False)
        total = grouped.ngroups
        checkpoint = max(1, total // 10)

        for idx, ((_, _att_id), group) in enumerate(grouped):
            label = _map_outcome(
                int(group["outcome_class"].iloc[0]),
                float(group["duration_days"].iloc[0]),
                str(group["attendance_type"].iloc[0]),
            )
            if label < 0:
                continue

            sequences.append(self._pad(group["_token_id"].tolist()))
            labels.append(label)
            hospital_ids.append(str(group["hospital_id"].iloc[0]))
            demographics.append([float(group["_age_norm"].iloc[0]), float(group["_sex_bin"].iloc[0])])

            if (idx + 1) % checkpoint == 0:
                pct = (idx + 1) / total * 100
                msg = f"[pipeline] tensores: {idx + 1:,}/{total:,} ({pct:.0f}%)"
                logger.info(msg)
                print(msg, flush=True)

        return (
            torch.tensor(sequences, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            hospital_ids,
            torch.tensor(demographics, dtype=torch.float32),
        )

    def _pad(self, tokens: List[int]) -> List[int]:
        tokens = tokens[: self.max_seq_len]
        return tokens + [0] * (self.max_seq_len - len(tokens))