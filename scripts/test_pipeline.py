"""
Diagnóstico do SequencePipeline — Abordagem de Tempo de Internação.

Uso:
    .venv/bin/python scripts/test_pipeline.py
    .venv/bin/python scripts/test_pipeline.py --db-url postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl
    .venv/bin/python scripts/test_pipeline.py --max-seq-len 64 --sample 5
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

LABEL_NAMES = {0: "curta (1-3d)", 1: "média (4-7d)", 2: "longa (8-14d)",
               3: "muito longa (15-30d)", 4: "prolongada (>30d)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-url",
        default=os.getenv("FL_DB_URL", "postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"),
        help="Connection string PostgreSQL",
    )
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument(
        "--sample", type=int, default=3,
        help="Número de sequências para imprimir como exemplo",
    )
    return p.parse_args()


_t_section: float = 0.0


def section(title: str) -> None:
    global _t_section
    _t_section = time.time()
    print(f"\n{'─' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * 60}", flush=True)


def section_done(extra: str = "") -> None:
    elapsed = time.time() - _t_section
    msg = f"  ✓ {elapsed:.1f}s"
    if extra:
        msg += f"  {extra}"
    print(msg, flush=True)


def main():
    args = parse_args()

    from src.mosaicfl.core.preprocessor import SequencePipeline

    print(flush=True)
    print("═" * 60, flush=True)
    print("  DIAGNÓSTICO — SequencePipeline (Tempo de Internação)", flush=True)
    print("═" * 60, flush=True)
    print(flush=True)
    print("  O que será feito:", flush=True)
    print("    1. Conectar ao PostgreSQL e executar query nos internados", flush=True)
    print("       de HSL e BPSP (3 tabelas: attendances, clinical_outcomes,", flush=True)
    print("       exam_records). Esta etapa pode levar vários minutos.", flush=True)
    print("    2. Construir vocabulário de analitos com bucket de valor", flush=True)
    print("       (baixo/normal/alto) baseado nos limites de referência.", flush=True)
    print("    3. Montar tensores de sequência temporal por paciente,", flush=True)
    print("       ordenados por dia_relativo (dias desde a admissão).", flush=True)
    print("    4. Validar compatibilidade com o modelo SimplifiedBEHRT.", flush=True)
    print(flush=True)
    print("  O que é esperado ao final:", flush=True)
    print("    • sequences : tensor (N, 128) — N atendimentos internados", flush=True)
    print("    • labels    : tensor (N,) com 5 classes de duração:", flush=True)
    print("                  0=curta 1=média 2=longa 3=muito longa 4=prolongada", flush=True)
    print("    • vocab     : dicionário com tokens do tipo 'leucocitos_alto'", flush=True)
    print("    • forward pass no BEHRT sem erros de shape", flush=True)
    print(flush=True)
    print(f"  Banco   : {args.db_url[:55]}...", flush=True)
    print(f"  seq_len : {args.max_seq_len} tokens por paciente", flush=True)
    print(f"  sample  : {args.sample} exemplos de sequência serão impressos", flush=True)
    print(flush=True)

    t_total = time.time()

    # ── 1. Build ──────────────────────────────────────────────────────────────
    section("1. Construindo sequências (os logs do pipeline aparecem abaixo)")
    pipeline = SequencePipeline(
        connection_string=args.db_url,
        max_seq_len=args.max_seq_len,
    )
    sequences, labels, vocab = pipeline.build()
    section_done(f"{len(sequences):,} pacientes/atendimentos carregados")

    # ── 2. Shape e tipos ──────────────────────────────────────────────────────
    section("2. Shapes e tipos")
    print(f"  sequences : {sequences.shape}  dtype={sequences.dtype}")
    print(f"  labels    : {labels.shape}     dtype={labels.dtype}")
    print(f"  vocab     : {len(vocab):,} tokens")
    section_done()

    # ── 3. Distribuição de classes ────────────────────────────────────────────
    section("3. Distribuição de labels (classes de duração)")
    total = len(labels)
    for cls in range(5):
        n = int((labels == cls).sum())
        pct = n / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls} — {LABEL_NAMES[cls]:<22}  {n:>6}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<28}  {total:>6}")
    section_done()

    # ── 4. Cobertura de tokens ────────────────────────────────────────────────
    section("4. Cobertura de tokens na sequência")
    pad_id = vocab["<PAD>"]
    unk_id = vocab["<UNK>"]

    non_pad = (sequences != pad_id).sum(dim=1).float()
    unk_count = (sequences == unk_id).sum(dim=1).float()

    print(f"  Comprimento real (tokens não-PAD):")
    print(f"    média   : {non_pad.mean():.1f}")
    print(f"    mediana : {non_pad.median():.1f}")
    print(f"    mín/máx : {non_pad.min():.0f} / {non_pad.max():.0f}")
    print(f"  Tokens <UNK> por sequência:")
    print(f"    média   : {unk_count.mean():.2f}")
    pct_unk = float((unk_count > 0).float().mean() * 100)
    print(f"    seq com pelo menos 1 UNK: {pct_unk:.1f}%")
    section_done()

    # ── 5. Vocabulário — top tokens ───────────────────────────────────────────
    section("5. Vocabulário — 20 tokens mais frequentes (exceto especiais)")
    especiais = {"<PAD>", "<UNK>", "<CLS>"}
    tokens_flat = sequences.flatten()
    id_to_token = {v: k for k, v in vocab.items()}

    from collections import Counter
    cnt = Counter(tokens_flat.tolist())
    for tok_id, count in cnt.most_common(25):
        token = id_to_token.get(tok_id, "?")
        if token in especiais:
            continue
        pct = count / tokens_flat.numel() * 100
        print(f"    {token:<35} id={tok_id:>5}  n={count:>8}  ({pct:.2f}%)")
    section_done()

    # ── 6. Exemplos de sequências ─────────────────────────────────────────────
    section(f"6. Exemplos de sequências ({args.sample} primeiras)")
    for i in range(min(args.sample, len(sequences))):
        seq = sequences[i]
        label = int(labels[i])
        real_tokens = [id_to_token.get(int(t), "?") for t in seq if int(t) != pad_id]
        print(f"\n  [Paciente {i}] label={label} ({LABEL_NAMES[label]})")
        print(f"  Tokens ({len(real_tokens)} reais):")
        for j, tok in enumerate(real_tokens[:12]):
            print(f"    pos {j:>3}  →  {tok}")
        if len(real_tokens) > 12:
            print(f"    ... +{len(real_tokens) - 12} tokens")
    section_done()

    # ── 7. Compatibilidade com o modelo ───────────────────────────────────────
    section("7. Compatibilidade com SimplifiedBEHRT")
    try:
        from src.mosaicfl.core.model import SimplifiedBEHRT
        from src.mosaicfl.core.config import MODEL_CFG

        if MODEL_CFG.num_classes != 5:
            print(f"  AVISO: MODEL_CFG.num_classes = {MODEL_CFG.num_classes}")
            print(f"  Este pipeline gera 5 classes. Atualize num_classes=5 em config.py")
            print(f"  antes de treinar. O teste abaixo usa num_classes atual.")

        model = SimplifiedBEHRT(use_cls_token=True)
        batch = sequences[:4]
        with torch.no_grad():
            logits = model(batch)
        print(f"  Forward pass OK — logits shape: {logits.shape}")
        print(f"  (batch=4, num_classes={MODEL_CFG.num_classes})")

    except Exception as e:
        print(f"  ERRO no forward pass: {e}")
    section_done()

    print(f"\n{'═' * 60}")
    print(f"  Pipeline OK — tempo total: {time.time() - t_total:.1f}s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
