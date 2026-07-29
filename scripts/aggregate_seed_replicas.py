"""
aggregate_seed_replicas.py — Agrega múltiplos treinamentos do Caminho B (rede
real), mesma configuração, sementes diferentes, em média/desvio-padrão/IC —
para reportar significância estatística sem depender de rodar dezenas de
réplicas (inviável na rede real, ver docs/Metodologia_Executada_MOSAIC-FL.tex,
seção Limitações).

FED_CFG.random_seed (FL_RANDOM_SEED) já é lido pelo treino local de cada
cliente também no Caminho B (src/mosaicfl/core/client.py::fit()) — o que
faltava era só esta agregação, análoga a `make seed-sensitivity` do Caminho A
(experiments/training_runner/run_seed_sensitivity.py), mas puxando resultados
já persistidos em metrics.fl_trainings em vez de orquestrar os treinos.

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    # rode os treinos (mesma config, FL_RANDOM_SEED diferente em cada máquina
    # a cada execução) e anote os training_ids retornados no log/API antes de
    # chamar este script:
    python3 scripts/aggregate_seed_replicas.py --training-ids 79 80 81
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

_COLUMNS = (
    "id, algorithm, run_classification, partition_mode, local_only_hospital, "
    "dp_noise_multiplier, dp_noise_strategy, best_accuracy, macro_f1, macro_auc, "
    "ece, ece_pre, best_round, n_rounds_done"
)

# Colunas de config que TODOS os training_ids precisam bater — se algum divergir,
# a agregação estatística mistura configurações diferentes e o resultado deixa
# de ser "variância por semente", vira "variância por configuração + semente".
_CONFIG_KEYS_MUST_MATCH = (
    "algorithm", "run_classification", "partition_mode",
    "local_only_hospital", "dp_noise_multiplier", "dp_noise_strategy",
)

_METRIC_KEYS = ("best_accuracy", "macro_f1", "macro_auc", "ece", "ece_pre")


def _confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """IC 95% aproximado via t-Student para amostras pequenas (n<30) — sem
    depender de scipy (não é dependência garantida em todo ambiente deste
    projeto). Tabela t crítica pros graus de liberdade mais comuns em réplicas
    de treino federado (poucas réplicas, custo alto por execução)."""
    _T_TABLE_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                   6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    n = len(values)
    if n < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    m = mean(values)
    sd = pstdev(values) * (n / (n - 1)) ** 0.5  # desvio-padrão amostral (n-1)
    df = min(n - 1, 10)
    t_crit = _T_TABLE_95.get(df, 1.96)  # normal como fallback pra n grande
    margin = t_crit * sd / (n ** 0.5)
    return (m - margin, m + margin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ids", type=int, nargs="+", required=True,
                         help="IDs de metrics.fl_trainings — mesma config, seeds diferentes")
    args = parser.parse_args()

    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)
    if len(args.training_ids) < 2:
        print("ERRO: precisa de pelo menos 2 training_ids pra calcular variância.")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = [
            dict(conn.execute(
                text(f"SELECT {_COLUMNS} FROM metrics.fl_trainings WHERE id = :id"),
                {"id": tid},
            ).mappings().first() or {})
            for tid in args.training_ids
        ]

    missing = [tid for tid, r in zip(args.training_ids, rows) if not r]
    if missing:
        print(f"ERRO: training_id(s) não encontrado(s): {missing}")
        sys.exit(1)

    print("=" * 70)
    print(f"RÉPLICAS DE SEMENTE — {len(rows)} execuções, Caminho B (rede real)")
    print("=" * 70)

    mismatches = []
    reference = rows[0]
    for key in _CONFIG_KEYS_MUST_MATCH:
        values = {r[key] for r in rows}
        if len(values) > 1:
            mismatches.append((key, values))

    if mismatches:
        print("\n>>> AVISO — configs divergem entre os training_ids informados:")
        for key, values in mismatches:
            print(f"    {key}: {values}")
        print("    A variância abaixo mistura configuração + semente — não é")
        print("    significância estatística válida. Confira os IDs.\n")
    else:
        print(f"\nConfig consistente entre as {len(rows)} execuções — variância "
              "abaixo reflete só a semente.\n")

    print(f"{'training_id':<12} {'best_round':<11} {'accuracy':<10} {'macro_f1':<10} "
          f"{'macro_auc':<10} {'ece_pre':<9} {'ece':<9}")
    for r in rows:
        print(f"{r['id']:<12} {r['best_round'] or '—':<11} "
              f"{fmt(r['best_accuracy']):<10} {fmt(r['macro_f1']):<10} "
              f"{fmt(r['macro_auc']):<10} {fmt(r['ece_pre']):<9} {fmt(r['ece']):<9}")

    print("\n" + "-" * 70)
    print(f"{'métrica':<12} {'média':<10} {'desvio-padrão':<15} {'IC 95%':<20}")
    for key in _METRIC_KEYS:
        values = [r[key] for r in rows if r[key] is not None]
        if len(values) < 2:
            print(f"{key:<12} {'n/a (< 2 valores)':<45}")
            continue
        m = mean(values)
        sd = pstdev(values) * (len(values) / (len(values) - 1)) ** 0.5
        lo, hi = _confidence_interval_95(values)
        print(f"{key:<12} {m:<10.4f} {sd:<15.4f} [{lo:.4f}, {hi:.4f}]")


def fmt(v) -> str:
    return "—" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
