"""
compare_two_proportions.py — Teste-z de duas proporções (p', 2026-07-30,
confirmado com a autora) entre dois treinamentos do Caminho B.

Serve pra duas perguntas diferentes, escolhidas por --class:
  (padrão, sem --class) compara a ACURÁCIA GERAL de dois treinos — ex.: "a
      diferença de accuracy entre o federado e o local-only é estatisticamente
      significativa, ou pode ser só ruído de amostra?"
  (com --class NOME)    compara o RECALL (ou --metric precision) de UMA
      classe específica entre dois treinos — a hipótese mais concreta que a
      autora tem hoje: "peso de classe mudou de forma significativa o recall
      de curado_internado, ou a diferença observada está dentro do esperado
      por acaso?"

Depende de confusion_matrix_stats já estar em evaluation_json (achado
2026-07-29) — só treinos rodados APÓS essa mudança de código têm esse campo.
Treinos antigos (id < ~79 nesta sessão) não têm e o script avisa.

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    # accuracy geral
    python3 scripts/compare_two_proportions.py --training-id-a 79 --training-id-b 80
    # recall de uma classe rara específica
    python3 scripts/compare_two_proportions.py --training-id-a 79 --training-id-b 80 \\
        --class curado_internado --metric recall
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from mosaicfl.core.confusion_stats import two_proportion_z_test

_METRIC_KEY = {"recall": "recall_sensitivity", "precision": "precision"}


def _load_confusion_stats(conn, training_id: int) -> dict | None:
    row = conn.execute(text(
        "SELECT evaluation_json FROM metrics.fl_checkpoints WHERE training_id = :id"
    ), {"id": training_id}).mappings().first()
    if not row or not row["evaluation_json"]:
        return None
    return row["evaluation_json"].get("confusion_matrix_stats")


def _proportion_and_n(stats: dict, class_name: str | None, metric: str) -> tuple[float, int]:
    if class_name is None:
        return stats["accuracy_p_hat"], stats["n_total"]
    per_class = stats["per_class_stats"].get(class_name)
    if per_class is None:
        raise ValueError(f"Classe '{class_name}' não encontrada. Disponíveis: {list(stats['per_class_stats'])}")
    value = per_class[_METRIC_KEY[metric]]
    support = per_class["support"]
    if value is None:
        raise ValueError(f"'{class_name}' tem suporte insuficiente pra {metric} nesse treino (support={support}).")
    return value, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-id-a", type=int, required=True)
    parser.add_argument("--training-id-b", type=int, required=True)
    parser.add_argument("--class", dest="class_name", default=None,
                         help="nome da classe (ex.: curado_internado) — default: accuracy geral")
    parser.add_argument("--metric", choices=["recall", "precision"], default="recall",
                         help="só usado com --class (default: recall/sensibilidade)")
    parser.add_argument("--alpha", type=float, default=0.05, help="nível de significância (default 0.05)")
    args = parser.parse_args()

    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        stats_a = _load_confusion_stats(conn, args.training_id_a)
        stats_b = _load_confusion_stats(conn, args.training_id_b)

    missing = []
    if stats_a is None:
        missing.append(args.training_id_a)
    if stats_b is None:
        missing.append(args.training_id_b)
    if missing:
        print(f"ERRO: training_id(s) sem confusion_matrix_stats em evaluation_json: {missing}")
        print("  Só treinos rodados com o código de 2026-07-29 em diante têm esse campo — "
              "treinos antigos não têm como ser reconstruídos (dado bruto nunca centralizado).")
        sys.exit(1)

    try:
        p_hat_a, n_a = _proportion_and_n(stats_a, args.class_name, args.metric)
        p_hat_b, n_b = _proportion_and_n(stats_b, args.class_name, args.metric)
    except ValueError as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    metric_label = "accuracy geral" if args.class_name is None else f"{args.metric} de '{args.class_name}'"
    print("=" * 70)
    print(f"TESTE-Z DE DUAS PROPORÇÕES — {metric_label}")
    print(f"training_id A={args.training_id_a} vs. B={args.training_id_b}")
    print("=" * 70)

    result = two_proportion_z_test(p_hat_a, n_a, p_hat_b, n_b, alpha=args.alpha)
    print(f"\n  p̂ A = {result['p_hat_1']:.4f}  (n={result['n1']})")
    print(f"  p̂ B = {result['p_hat_2']:.4f}  (n={result['n2']})")
    print(f"  Diferença (A − B): {result['diff']:+.4f}")
    print(f"\n  p' (proporção agrupada): {result['p_pooled']:.4f}")
    print(f"  Estatística z: {result['z_statistic']:.4f}")
    print(f"  p-valor (bicaudal): {result['p_value']:.6f}")
    print(f"\n  >>> {'ESTATISTICAMENTE SIGNIFICATIVO' if result['significant'] else 'NÃO significativo'} "
          f"a α={args.alpha} — {'rejeita' if result['significant'] else 'não rejeita'} H0 "
          f"(as duas proporções vêm da mesma população).")


if __name__ == "__main__":
    main()
