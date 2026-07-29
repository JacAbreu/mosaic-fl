"""
post_training_hooks.py — Roda depois de `make server-app` (Caminho B, rede
real) para fechar os itens 3 e 4 da lista de lacunas automaticamente:

  3. Gera amostras de avaliação do RAG (chama scripts/rag_likert_evaluation.py
     --generate) — nota humana continua pendente (rode --score-pending depois,
     no seu ritmo), mas a nota automática (llm_judge_score) já fica pronta.
  4. Detecta se já existem réplicas da mesma configuração (mesma algorithm,
     run_classification, partition_mode, local_only_hospital, dp_noise_*) e,
     se houver 2+, roda a agregação de significância estatística
     (scripts/aggregate_seed_replicas.py). Com menos de 2, só avisa.

`make server-app` (flwr run . production) SUBMETE o treino e retorna quase
imediatamente — o treino em si roda de forma assíncrona no ServerApp. Por
isso este script primeiro ESPERA (poll no banco) o treino mais recentemente
registrado terminar antes de rodar as avaliações.

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    python3 scripts/post_training_hooks.py                    # detecta o treino mais recente
    python3 scripts/post_training_hooks.py --training-id 79   # explícito
    python3 scripts/post_training_hooks.py --skip-rag         # só significância
    python3 scripts/post_training_hooks.py --skip-significance  # só RAG
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

_CONFIG_KEYS = (
    "algorithm", "run_classification", "partition_mode",
    "local_only_hospital", "dp_noise_multiplier", "dp_noise_strategy",
)


def _most_recent_training_id(conn) -> int | None:
    row = conn.execute(text(
        "SELECT id FROM metrics.fl_trainings ORDER BY id DESC LIMIT 1"
    )).fetchone()
    return row.id if row else None


def _wait_for_completion(conn, training_id: int, timeout_s: int, poll_s: int) -> str:
    """Poll em metrics.fl_trainings.status até 'completed'/'failed' ou timeout.
    Retorna o status final observado ('completed', 'failed', ou 'timeout')."""
    waited = 0
    while waited < timeout_s:
        row = conn.execute(text(
            "SELECT status FROM metrics.fl_trainings WHERE id = :id"
        ), {"id": training_id}).fetchone()
        if row is None:
            return "not_found"
        if row.status in ("completed", "failed"):
            return row.status
        time.sleep(poll_s)
        waited += poll_s
        print(f"  ... training_id={training_id} ainda '{row.status}' "
              f"({waited}s / {timeout_s}s de espera)")
    return "timeout"


def _find_sibling_ids(conn, training_id: int) -> list[int]:
    """training_ids completados com a MESMA config-chave do training_id dado
    (achado 2026-07-29) — candidatos a réplicas de semente pra significância
    estatística. Não valida se a semente (FL_RANDOM_SEED) de fato variou —
    só confirma que a config declarada bate; a variação real vem de fora
    (ela roda o treino de novo mudando a semente manualmente)."""
    ref = conn.execute(text(
        f"SELECT id, {', '.join(_CONFIG_KEYS)} FROM metrics.fl_trainings WHERE id = :id"
    ), {"id": training_id}).mappings().first()
    if not ref:
        return []

    where_clauses = []
    params: dict = {"id": training_id}
    for key in _CONFIG_KEYS:
        if ref[key] is None:
            where_clauses.append(f"{key} IS NULL")
        else:
            where_clauses.append(f"{key} = :{key}")
            params[key] = ref[key]

    rows = conn.execute(text(
        f"SELECT id FROM metrics.fl_trainings "
        f"WHERE status = 'completed' AND {' AND '.join(where_clauses)} "
        "ORDER BY id"
    ), params).fetchall()
    return [r.id for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-id", type=int, default=None,
                         help="ID a esperar (default: o mais recente registrado no banco)")
    parser.add_argument("--timeout-s", type=int, default=6 * 3600, help="timeout de espera, segundos (default 6h)")
    parser.add_argument("--poll-s", type=int, default=60, help="intervalo de poll, segundos (default 60)")
    parser.add_argument("--rag-n", type=int, default=20, help="nº de amostras pro RAG (default 20)")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--skip-significance", action="store_true")
    args = parser.parse_args()

    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        training_id = args.training_id or _most_recent_training_id(conn)
        if training_id is None:
            print("ERRO: nenhum treino encontrado em metrics.fl_trainings.")
            sys.exit(1)

        print(f"Aguardando conclusão do training_id={training_id}...")
        status = _wait_for_completion(conn, training_id, args.timeout_s, args.poll_s)

    if status != "completed":
        print(f"\nERRO: training_id={training_id} terminou com status='{status}' "
              "(esperado 'completed') — hooks de avaliação NÃO disparados.")
        sys.exit(1)

    print(f"\ntraining_id={training_id} concluído. Rodando hooks pós-treino.\n")

    if not args.skip_significance:
        print("=" * 70)
        print("ITEM 4 — significância estatística (réplicas de semente)")
        print("=" * 70)
        with engine.connect() as conn:
            siblings = _find_sibling_ids(conn, training_id)
        if len(siblings) >= 2:
            print(f"Encontradas {len(siblings)} execuções com a mesma config: {siblings}")
            subprocess.run(
                [sys.executable, "scripts/aggregate_seed_replicas.py",
                 "--training-ids", *[str(i) for i in siblings]],
                check=False,
            )
        else:
            print(f"Só {len(siblings)} execução(ões) com essa config até agora "
                  f"({siblings}). Rode de novo com FL_RANDOM_SEED diferente pra "
                  "ter uma 2ª réplica e calcular significância.")

    if not args.skip_rag:
        print("\n" + "=" * 70)
        print("ITEM 3 — avaliação do RAG (geração automática, nota humana pendente)")
        print("=" * 70)
        subprocess.run(
            [sys.executable, "scripts/rag_likert_evaluation.py", "--generate", "--n", str(args.rag_n)],
            check=False,
        )
        print("\nAmostras geradas com nota automática (llm_judge_score). Pra dar SUA "
              "nota humana: python3 scripts/rag_likert_evaluation.py --score-pending")


if __name__ == "__main__":
    main()
