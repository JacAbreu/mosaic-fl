"""
compare_local_vs_federated.py — Compara o treinamento federado mais recente
(Caminho B, rede real) contra os baselines locais isolados (BPSP-only,
HSL-only) mais recentes, também no Caminho B.

Achado 2026-07-28: a comparação local x federado só existia entre pipelines
diferentes (baseline local do Caminho A vs. federado do Caminho B) ou estava
desatualizada (T13-T15, anterior a praticamente todas as correções desta
sessão). Agora que treinos local-only rodam na própria rede real
(metrics.fl_trainings.local_only_hospital, migration 030), a comparação passa
a ser "maçã com maçã" — mesma arquitetura, mesmo pipeline, mesma máquina.

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    python3 scripts/compare_local_vs_federated.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

_SUMMARY_COLUMNS = (
    "id, algorithm, status, started_at, completed_at, n_rounds_done, best_round, "
    "best_accuracy, macro_f1, macro_auc, ece_pre, ece, local_only_hospital, "
    "rag_precision_at_k, dp_noise_multiplier"
)


def _latest(conn, local_only_hospital):
    if local_only_hospital is None:
        where = "local_only_hospital IS NULL"
        params = {}
    else:
        where = "local_only_hospital = :hospital"
        params = {"hospital": local_only_hospital}
    row = conn.execute(text(
        f"SELECT {_SUMMARY_COLUMNS} FROM metrics.fl_trainings "
        f"WHERE status = 'completed' AND {where} "
        "ORDER BY id DESC LIMIT 1"
    ), params).mappings().first()
    return dict(row) if row else None


def _per_class_f1_at_best_round(conn, training_id, best_round):
    if training_id is None or best_round is None:
        return None
    row = conn.execute(text(
        "SELECT per_class_f1 FROM metrics.fl_round_history "
        "WHERE training_id = :tid AND round = :rnd"
    ), {"tid": training_id, "rnd": best_round}).mappings().first()
    return row["per_class_f1"] if row else None


def _print_side(title, row, per_class_f1, class_labels):
    print(f"\n{title}")
    print("-" * len(title))
    if row is None:
        print("  Nenhum treino completo encontrado com esse escopo.")
        return
    print(f"  training_id:      {row['id']}")
    print(f"  algorithm:        {row['algorithm']}")
    print(f"  n_rounds_done:    {row['n_rounds_done']}")
    print(f"  best_round:       {row['best_round']}")
    print(f"  best_accuracy:    {row['best_accuracy']}")
    print(f"  macro_f1:         {row['macro_f1']}")
    print(f"  macro_auc:        {row['macro_auc']}")
    print(f"  ece (pré → pós):  {row['ece_pre']} → {row['ece']}")
    print(f"  rag_precision_at_k: {row['rag_precision_at_k']}")
    print(f"  dp_noise_multiplier: {row['dp_noise_multiplier']}")
    if per_class_f1:
        print("  per_class_f1 (na melhor rodada):")
        for lbl, f1 in zip(class_labels, per_class_f1):
            print(f"    {lbl:<28} {f1:.4f}")


def main() -> None:
    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    from mosaicfl.core.config import MODEL_CFG
    class_labels = list(MODEL_CFG.class_labels)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        federated = _latest(conn, None)
        bpsp_only = _latest(conn, "BPSP")
        hsl_only  = _latest(conn, "HSL")

        fed_pcf1  = _per_class_f1_at_best_round(conn, federated["id"], federated["best_round"]) if federated else None
        bpsp_pcf1 = _per_class_f1_at_best_round(conn, bpsp_only["id"], bpsp_only["best_round"]) if bpsp_only else None
        hsl_pcf1  = _per_class_f1_at_best_round(conn, hsl_only["id"], hsl_only["best_round"]) if hsl_only else None

    print("=" * 70)
    print("COMPARAÇÃO LOCAL x FEDERADO — Caminho B (rede real)")
    print("=" * 70)
    print(
        "\nAVISO: comparação só é válida se os três treinos usarem a mesma\n"
        "arquitetura/hiperparâmetros (peso de classe, DP, etc.) — confira as\n"
        "datas (started_at) e configs antes de citar como resultado formal."
    )

    _print_side("FEDERADO (BPSP + HSL, mais recente)", federated, fed_pcf1, class_labels)
    _print_side("LOCAL — BPSP sozinho (mais recente)", bpsp_only, bpsp_pcf1, class_labels)
    _print_side("LOCAL — HSL sozinho (mais recente)", hsl_only, hsl_pcf1, class_labels)

    if federated and bpsp_only and hsl_only:
        print("\n" + "=" * 70)
        print("EFEITO EQUALIZADOR DO FL — ganho do federado sobre cada local")
        print("=" * 70)
        for name, local_row, local_pcf1 in [("BPSP", bpsp_only, bpsp_pcf1), ("HSL", hsl_only, hsl_pcf1)]:
            delta_acc = federated["best_accuracy"] - local_row["best_accuracy"]
            delta_f1  = federated["macro_f1"] - local_row["macro_f1"]
            print(f"\n  Federado vs. {name} local:")
            print(f"    Δ accuracy: {delta_acc:+.4f}")
            print(f"    Δ macro_f1: {delta_f1:+.4f}")
            if fed_pcf1 and local_pcf1:
                print("    Δ per_class_f1 (federado − local):")
                for lbl, f_fed, f_local in zip(class_labels, fed_pcf1, local_pcf1):
                    print(f"      {lbl:<28} {f_fed - f_local:+.4f}")


if __name__ == "__main__":
    main()
