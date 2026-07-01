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
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import MODEL_CFG
from ..model import MAX_DIA_RELATIVO
from .outcomes import _map_outcome
from .tokens import TokenMode

logger = logging.getLogger(__name__)


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


class SequencePipeline:
    _SPECIAL: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}

    # Ano de referência do dataset FAPESP COVID-19 (internações 2020-2021).
    # Usado para calcular age_at_admission = _FAPESP_REF_YEAR - birth_year.
    _FAPESP_REF_YEAR: int = 2021

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
        sequences, labels, _, demographics, dia_relativos = self._build_tensors(df, vocab)
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
        return sequences, labels, vocab, demographics, dia_relativos

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
            seqs, lbls, _, demo, dia_rels = self._build_tensors(df_h, vocab)
            dist = {i: int((lbls == i).sum()) for i in range(MODEL_CFG.num_classes)}
            _log(
                f"[pipeline/per_hospital] {hosp}: {len(seqs):,} sequências "
                f"em {time.time() - t_h:.1f}s | dist={dist} | "
                f"age_mean={demo[:, 0].mean():.2f} sex_M={int((demo[:, 1] == 1.0).sum())}"
            )
            result[hosp] = (seqs, lbls, vocab, demo, dia_rels)

        _log(f"[pipeline/per_hospital] concluído em {time.time() - t0:.1f}s total")
        return result

    def _load_dataframe(self) -> pd.DataFrame:
        """Conecta ao banco, executa a query principal e retorna o DataFrame bruto."""
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

    def _build_tensors(
        self, df: pd.DataFrame, vocab: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
        """
        Retorna (sequences, labels, hospital_ids, demographics, dia_relativos).

        Token building e demográficos são computados vetorizadamente no DataFrame
        inteiro antes do groupby — elimina iterrows() no inner loop.

        demographics: FloatTensor shape (N, 2) — [age_norm, sex_binary] por paciente.
            age_norm   = (REF_YEAR − birth_year) / 100.0, clamped [0.0, 1.0]
                         0.5 quando birth_year ausente
            sex_binary = 1.0 para 'M', 0.0 para 'F' ou desconhecido
        dia_relativos: LongTensor shape (N, max_seq_len) — dias desde admissão por token.
            0 = padding; dia 0 (admissão) → índice 1; dia ≥ MAX_DIA_RELATIVO → MAX_DIA_RELATIVO+1
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

        # ── Vectorized dia_relativo (temporal position within episode) ─────────
        if "dia_relativo" in df.columns:
            # Shift +1: 0=padding, 1=dia0 (admissão), ..., MAX_DIA_RELATIVO+1=dia≥MAX
            dia_shifted = (
                pd.to_numeric(df["dia_relativo"], errors="coerce")
                .fillna(0)
                .clip(upper=MAX_DIA_RELATIVO - 1)
                .astype(int)
                + 1
            ).values
        else:
            dia_shifted = np.ones(len(df), dtype=int)  # fallback: todos no dia 1
        df = df.assign(_dia_rel=dia_shifted)

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
        dia_relativos: List[List[int]] = []

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
            dia_relativos.append(self._pad(group["_dia_rel"].tolist()))

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
            torch.tensor(dia_relativos, dtype=torch.long),
        )

    def _pad(self, tokens: List[int]) -> List[int]:
        tokens = tokens[: self.max_seq_len]
        return tokens + [0] * (self.max_seq_len - len(tokens))
