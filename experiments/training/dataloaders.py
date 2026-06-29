"""Preparação de DataLoaders para FL — modo sintético/CSV e modo banco de dados real."""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from mosaicfl.core.config import BATCH_SIZE, MAX_SEQ_LEN, NUM_CLIENTS, RANDOM_SEED
from mosaicfl.core.preprocessor import EHRPreprocessor, SequencePipeline, split_by_institution

logger = logging.getLogger(__name__)


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


def prepare_dataloaders_from_db(
    db_url: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[Dict, DataLoader, Dict, int, Dict, DataLoader, DataLoader]:
    """
    Constrói DataLoaders para FL a partir dos tensores reais do SequencePipeline.

    Divide por hospital: cada hospital vira um cliente FL.
    Split: 70% treino / 10% val / 10% cal (calibração) / 10% teste global.
    cal é reservado exclusivamente para temperature scaling — nunca exposto ao FL.

    FL_INCLUDE_HOSPITALS (CSV, ex: "BPSP" ou "HSL,BPSP"):
        Se definido, apenas os hospitais listados entram como clientes de treino.
        O test set e cal set continuam globais (todos os hospitais) para comparação justa.
        Uso: leave-one-client-out — permite isolar a contribuição de cada hospital.

    Returns:
        client_loaders, test_loader, vocab, total_train_samples,
        demographics_by_client, test_loader_demo, cal_loader
    """
    import os
    _include_raw = os.getenv("FL_INCLUDE_HOSPITALS", "").strip()
    include_hospitals = (
        {h.strip() for h in _include_raw.split(",") if h.strip()}
        if _include_raw else None
    )
    if include_hospitals:
        logger.info("[db] FL_INCLUDE_HOSPITALS=%s — leave-one-client-out mode", sorted(include_hospitals))
    else:
        logger.info("[db] FL_INCLUDE_HOSPITALS não definido — todos os hospitais como clientes")

    logger.info("[db] Iniciando carregamento via SequencePipeline...")
    pipeline = SequencePipeline(connection_string=db_url, max_seq_len=MAX_SEQ_LEN)
    hospital_data = pipeline.build_per_hospital()

    if not hospital_data:
        raise RuntimeError("build_per_hospital() retornou vazio — sem dados na base.")

    vocab = next(iter(hospital_data.values()))[2]

    rng = torch.Generator()
    rng.manual_seed(RANDOM_SEED)

    client_loaders: Dict = {}
    demographics_by_client: Dict = {}
    test_seqs_list, test_lbls_list, test_demo_list, test_dia_list = [], [], [], []
    cal_seqs_list, cal_lbls_list, cal_dia_list = [], [], []
    total_train_samples = 0

    for cid, (hospital_id, (seqs, labels, _, demo, dia_rels)) in enumerate(hospital_data.items()):
        n = len(seqs)
        if n < 10:
            logger.warning(f"[db] Hospital {hospital_id}: apenas {n} amostras — pulando.")
            continue

        perm = torch.randperm(n, generator=rng)
        n_train = int(0.7 * n)
        n_val   = int(0.1 * n)
        n_cal   = int(0.1 * n)

        train_seqs = seqs[perm[:n_train]]
        train_lbls = labels[perm[:n_train]]
        train_demo = demo[perm[:n_train]]
        train_dia  = dia_rels[perm[:n_train]]
        val_seqs   = seqs[perm[n_train:n_train + n_val]]
        val_lbls   = labels[perm[n_train:n_train + n_val]]
        val_demo   = demo[perm[n_train:n_train + n_val]]
        val_dia    = dia_rels[perm[n_train:n_train + n_val]]
        cal_seqs   = seqs[perm[n_train + n_val:n_train + n_val + n_cal]]
        cal_lbls   = labels[perm[n_train + n_val:n_train + n_val + n_cal]]
        cal_dia    = dia_rels[perm[n_train + n_val:n_train + n_val + n_cal]]

        # test e cal são sempre globais — independente do filtro de cliente
        test_seqs_list.append(seqs[perm[n_train + n_val + n_cal:]])
        test_lbls_list.append(labels[perm[n_train + n_val + n_cal:]])
        test_demo_list.append(demo[perm[n_train + n_val + n_cal:]])
        test_dia_list.append(dia_rels[perm[n_train + n_val + n_cal:]])
        cal_seqs_list.append(cal_seqs)
        cal_lbls_list.append(cal_lbls)
        cal_dia_list.append(cal_dia)

        if include_hospitals is not None and hospital_id not in include_hospitals:
            logger.info(
                "[db] Hospital %s → excluído do treino (FL_INCLUDE_HOSPITALS); test/cal mantidos globais",
                hospital_id,
            )
            continue

        # gerador fixo por cliente — shuffling determinístico, reproduzível entre runs
        _gen = torch.Generator().manual_seed(RANDOM_SEED + cid)
        client_loaders[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls, train_dia), batch_size=batch_size, shuffle=True, generator=_gen),
            DataLoader(TensorDataset(val_seqs, val_lbls, val_dia), batch_size=batch_size),
        )
        _gen_demo = torch.Generator().manual_seed(RANDOM_SEED + cid)
        demographics_by_client[cid] = (
            DataLoader(TensorDataset(train_seqs, train_lbls, train_demo, train_dia), batch_size=batch_size, shuffle=True, generator=_gen_demo),
            DataLoader(TensorDataset(val_seqs, val_lbls, val_demo, val_dia), batch_size=batch_size),
        )
        total_train_samples += len(train_seqs)
        logger.info(
            f"[db] Hospital {hospital_id} → cliente {cid}: "
            f"{len(train_seqs)} treino | {len(val_seqs)} val | {len(cal_seqs)} cal | "
            f"age_mean={train_demo[:, 0].mean():.2f} sex_M={int((train_demo[:, 1] == 1.0).sum())}"
        )

    if not client_loaders:
        raise RuntimeError("Nenhum cliente válido criado a partir dos dados reais.")

    test_seqs  = torch.cat(test_seqs_list, dim=0)
    test_lbls  = torch.cat(test_lbls_list, dim=0)
    test_demo  = torch.cat(test_demo_list, dim=0)
    test_dia   = torch.cat(test_dia_list, dim=0)
    cal_seqs_all = torch.cat(cal_seqs_list, dim=0)
    cal_lbls_all = torch.cat(cal_lbls_list, dim=0)
    cal_dia_all  = torch.cat(cal_dia_list, dim=0)

    test_loader      = DataLoader(TensorDataset(test_seqs, test_lbls, test_dia), batch_size=batch_size)
    test_loader_demo = DataLoader(TensorDataset(test_seqs, test_lbls, test_demo, test_dia), batch_size=batch_size)
    cal_loader       = DataLoader(TensorDataset(cal_seqs_all, cal_lbls_all, cal_dia_all), batch_size=batch_size)

    logger.info(
        f"[db] Teste global: {len(test_seqs)} amostras | Cal global: {len(cal_seqs_all)} amostras "
        f"| {len(client_loaders)} clientes FL"
    )

    return client_loaders, test_loader, vocab, total_train_samples, demographics_by_client, test_loader_demo, cal_loader
