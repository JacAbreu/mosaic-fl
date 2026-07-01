"""
legacy_reference.py — Abordagem inicial de construção de sequências, preservada como referência.

Nenhum código do projeto usa SequencePipelineInicial em produção ou simulação —
ver SequencePipeline (sequence_pipeline.py) para o pipeline real.
"""
import logging
from typing import List, Tuple

import pandas as pd
import torch

from ..config import MODEL_CFG

logger = logging.getLogger(__name__)


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
