"""
rag_likert_evaluation.py — Avaliação da justificativa gerada pelo RAG, sobre
pacientes reais locais deste hospital. Fecha o Experimento 4 do plano de
metodologia original (nunca executado até 2026-07-28) — ver
docs/Comparativo_Metodologia_Planejada_vs_Executada.md, Seção 10.

MEDE A QUALIDADE DA JUSTIFICATIVA GERADA (texto claro/correto?) — diferente
de Precision@k (mede se os casos RECUPERADOS são da classe certa; ver
mosaicfl.core.rag.precision). As duas métricas são complementares, não
substitutas uma da outra.

Duas notas são registradas por amostra, e NUNCA devem ser confundidas:
  likert_score     — HUMANA. Só você atribui (ou quem você designar como
                      avaliador). Nunca inferida automaticamente.
  llm_judge_score  — AUTOMÁTICA (LLM-como-juiz, achado 2026-07-29, decisão
                      explícita via AskUserQuestion). Métrica COMPLEMENTAR,
                      calculada sem envolvimento humano — não é, e não deve
                      ser citada como, "avaliação humana".

Modos:
  (default)        gera amostras novas E pausa pra você dar a nota humana em
                    cada uma, na hora — fluxo de quando você quer avaliar
                    tudo numa sentada só.
  --generate        gera amostras novas (chama a API, calcula llm_judge_score),
                    persiste com likert_score=NULL — SEM pausar. Uso: chamado
                    automaticamente pelos alvos de treino no Makefile.
  --score-pending    revisa amostras já geradas com likert_score ainda NULL —
                    uso: você avaliando no seu ritmo, depois do --generate
                    ter rodado sozinho.
  --report          só imprime o resumo agregado (as duas métricas lado a
                    lado, com concordância entre elas quando aplicável).

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    python3 scripts/rag_likert_evaluation.py --generate --n 20
    python3 scripts/rag_likert_evaluation.py --score-pending --evaluator "Jacqueline"
    python3 scripts/rag_likert_evaluation.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from infrastructure.mosaicfl_api.audit import pseudonymize

_MIN_EXAMS_PER_PATIENT = 2


def _sample_patients(conn, n: int, seed: float) -> list[str]:
    conn.execute(text("SELECT setseed(:seed)"), {"seed": seed})
    rows = conn.execute(text("""
        SELECT patient_id FROM (
            SELECT patient_id, count(*) AS n_exams
            FROM metrics.exam_records
            GROUP BY patient_id
            HAVING count(*) >= :min_exams
        ) t
        ORDER BY random()
        LIMIT :n
    """), {"min_exams": _MIN_EXAMS_PER_PATIENT, "n": n}).fetchall()
    return [r.patient_id for r in rows]


def _load_exams(conn, patient_id: str) -> list[dict]:
    rows = conn.execute(text("""
        SELECT exam_name, date, value, phase, ref_low, ref_high
        FROM metrics.exam_records
        WHERE patient_id = :patient_id
        ORDER BY date
    """), {"patient_id": patient_id}).fetchall()
    return [
        {
            "exam_name": r.exam_name, "date": str(r.date), "value": r.value,
            "phase": r.phase, "ref_low": r.ref_low, "ref_high": r.ref_high,
        }
        for r in rows
    ]


def _call_predict(api_url: str, token: str | None, patient_id: str, exams: list[dict]) -> dict:
    """POST /api/predict via urllib (stdlib) — evita depender de requests/httpx
    como dependência nova só pra este script."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps({"patient_id": patient_id, "exams": exams}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url}/api/predict?explain=true", data=body, headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _prompt_likert() -> int | None:
    while True:
        raw = input("\nNota Likert (1=inútil/alucinação .. 5=totalmente clara e correta, "
                    "'s'=pular, 'q'=parar): ").strip().lower()
        if raw == "q":
            return -1
        if raw == "s":
            return None
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print("  Entrada inválida — digite 1-5, 's' ou 'q'.")


def _get_judge():
    """ClinicalRAG construído uma única vez (carregar o LLM é caro) — reaproveitado
    pra julgar todas as amostras da sessão."""
    from mosaicfl.core.rag import ClinicalRAG
    return ClinicalRAG()


def run_evaluation(args, pause_for_human: bool) -> None:
    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    judge = _get_judge()

    with engine.connect() as conn:
        patients = _sample_patients(conn, args.n, args.seed)
        print(f"Amostrados {len(patients)} pacientes reais (min. {_MIN_EXAMS_PER_PATIENT} exames cada).\n")

        n_generated = 0
        for i, patient_id in enumerate(patients, start=1):
            exams = _load_exams(conn, patient_id)
            print("=" * 70)
            print(f"[{i}/{len(patients)}] Paciente (hash): {pseudonymize(patient_id)}")
            print(f"  Exames ({len(exams)}): " + ", ".join(
                f"{e['exam_name']}={e['value']}" for e in exams[:8]
            ) + (" ..." if len(exams) > 8 else ""))

            try:
                result = _call_predict(args.api_url, args.token, patient_id, exams)
            except Exception as exc:
                print(f"  ERRO ao chamar /api/predict: {exc} — pulando este paciente.")
                continue

            rag = result.get("rag_explanation") or {}
            print(f"\n  Classe prevista: {result['predicted_label']} "
                  f"(risk_score={result['risk_score']:.4f})")
            if rag.get("erro"):
                print(f"  RAG: sem justificativa ({rag['erro']}) — pulando este paciente.")
                continue
            justificativa = rag.get("justificativa", "")
            print(f"  Justificativa: {justificativa or '(vazia)'}")
            print(f"  Confiável={rag.get('confiavel')} | "
                  f"Alucinação detectada={rag.get('alucinacao_detectada')} | "
                  f"backend={rag.get('llm_backend')} modelo={rag.get('llm_model_used')} "
                  f"fallback={rag.get('llm_was_fallback')}")

            judge_score, judge_rationale = judge.judge_justification(
                result["predicted_label"], justificativa, rag.get("fontes", []),
            )
            print(f"  [automático, não-humano] llm_judge_score={judge_score} "
                  f"({judge_rationale[:80] if judge_rationale else ''})")

            score = None
            if pause_for_human:
                score = _prompt_likert()
                if score == -1:
                    print("\nParando — amostras já registradas foram salvas.")
                    break
                if score is None:
                    print("  Pulado, não registrado (nem a nota automática).")
                    continue

            with engine.begin() as tx:
                tx.execute(text("""
                    INSERT INTO metrics.rag_likert_evaluations
                        (patient_id_hash, predicted_label, risk_score, justificativa,
                         fontes_json, llm_backend, llm_model_used, llm_was_fallback,
                         alucinacao_detectada, confiavel, likert_score, evaluator,
                         checkpoint_round, llm_judge_score, llm_judge_rationale,
                         llm_judge_backend, llm_judge_model)
                    VALUES
                        (:patient_id_hash, :predicted_label, :risk_score, :justificativa,
                         cast(:fontes_json as jsonb), :llm_backend, :llm_model_used, :llm_was_fallback,
                         :alucinacao_detectada, :confiavel, :likert_score, :evaluator,
                         :checkpoint_round, :llm_judge_score, :llm_judge_rationale,
                         :llm_judge_backend, :llm_judge_model)
                """), {
                    "patient_id_hash": pseudonymize(patient_id),
                    "predicted_label": result["predicted_label"],
                    "risk_score": result["risk_score"],
                    "justificativa": justificativa,
                    "fontes_json": json.dumps(rag.get("fontes", [])),
                    "llm_backend": rag.get("llm_backend"),
                    "llm_model_used": rag.get("llm_model_used"),
                    "llm_was_fallback": bool(rag.get("llm_was_fallback", False)),
                    "alucinacao_detectada": bool(rag.get("alucinacao_detectada", False)),
                    "confiavel": bool(rag.get("confiavel", False)),
                    "likert_score": score,
                    "evaluator": args.evaluator,
                    "checkpoint_round": (result.get("model_metadata") or {}).get("checkpoint_round"),
                    "llm_judge_score": judge_score,
                    "llm_judge_rationale": judge_rationale,
                    "llm_judge_backend": rag.get("llm_backend"),
                    "llm_judge_model": rag.get("llm_model_used"),
                })
            n_generated += 1

        modo = "avaliação(ões) humana(s)" if pause_for_human else "amostra(s) gerada(s) (nota humana pendente)"
        print(f"\n{n_generated} {modo} registrada(s) nesta sessão.")


def run_score_pending(args) -> None:
    """Revisa amostras já geradas (--generate) com likert_score ainda NULL —
    pra você avaliar no seu ritmo, sem precisar rechamar a API."""
    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, predicted_label, risk_score, justificativa, alucinacao_detectada,
                   confiavel, llm_judge_score, llm_judge_rationale
            FROM metrics.rag_likert_evaluations
            WHERE likert_score IS NULL
            ORDER BY id
        """)).fetchall()

    if not rows:
        print("Nenhuma amostra pendente de nota humana.")
        return

    print(f"{len(rows)} amostra(s) pendente(s) de avaliação humana.\n")
    n_scored = 0
    for i, r in enumerate(rows, start=1):
        print("=" * 70)
        print(f"[{i}/{len(rows)}] id={r.id} — classe prevista: {r.predicted_label} "
              f"(risk_score={r.risk_score:.4f})")
        print(f"  Justificativa: {r.justificativa}")
        print(f"  Alucinação detectada={r.alucinacao_detectada} | Confiável={r.confiavel}")
        print(f"  [referência automática, não-humana] llm_judge_score={r.llm_judge_score} "
              f"({r.llm_judge_rationale or ''})")

        score = _prompt_likert()
        if score == -1:
            print("\nParando.")
            break
        if score is None:
            print("  Pulado.")
            continue

        with engine.begin() as tx:
            tx.execute(text(
                "UPDATE metrics.rag_likert_evaluations SET likert_score = :score, "
                "evaluator = :evaluator WHERE id = :id"
            ), {"score": score, "evaluator": args.evaluator, "id": r.id})
        n_scored += 1

    print(f"\n{n_scored} avaliação(ões) humana(s) registrada(s) nesta sessão.")


def run_report(args) -> None:
    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    base_where = "1=1"
    params: dict = {}
    if args.evaluator:
        base_where += " AND evaluator = :evaluator"
        params["evaluator"] = args.evaluator

    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT likert_score, llm_judge_score, alucinacao_detectada, confiavel "
            f"FROM metrics.rag_likert_evaluations WHERE {base_where}"
        ), params).fetchall()

    if not rows:
        print("Nenhuma amostra registrada ainda (com esse filtro).")
        return

    n_total = len(rows)
    n_hallucination = sum(1 for r in rows if r.alucinacao_detectada)
    n_confiavel = sum(1 for r in rows if r.confiavel)

    print("=" * 60)
    print(f"AVALIAÇÃO DO RAG — resumo ({n_total} amostras geradas)")
    print("=" * 60)
    print(f"\nFrequência de alucinação detectada: {100 * n_hallucination / n_total:.1f}% ({n_hallucination}/{n_total})")
    print(f"Frequência marcada como confiável (rag.confiavel): {100 * n_confiavel / n_total:.1f}% ({n_confiavel}/{n_total})")

    human = [r for r in rows if r.likert_score is not None]
    print(f"\n--- Nota HUMANA (likert_score) — {len(human)}/{n_total} avaliadas ---")
    if human:
        n_useful = sum(1 for r in human if r.likert_score >= 4)
        dist = Counter(r.likert_score for r in human)
        for score in range(1, 6):
            bar = "█" * dist.get(score, 0)
            print(f"  {score}: {dist.get(score, 0):>3}  {bar}")
        print(f"  % com nota >= 4 (métrica do plano original, Experimento 4): "
              f"{100 * n_useful / len(human):.1f}% ({n_useful}/{len(human)})")
    else:
        print("  Nenhuma ainda — rode --score-pending pra avaliar as amostras geradas.")

    judged = [r for r in rows if r.llm_judge_score is not None]
    print(f"\n--- Nota AUTOMÁTICA (llm_judge_score, NÃO é avaliação humana) — "
          f"{len(judged)}/{n_total} ---")
    if judged:
        n_useful_judge = sum(1 for r in judged if r.llm_judge_score >= 4)
        dist_judge = Counter(r.llm_judge_score for r in judged)
        for score in range(1, 6):
            bar = "█" * dist_judge.get(score, 0)
            print(f"  {score}: {dist_judge.get(score, 0):>3}  {bar}")
        print(f"  % com nota >= 4: {100 * n_useful_judge / len(judged):.1f}% ({n_useful_judge}/{len(judged)})")

    both = [r for r in rows if r.likert_score is not None and r.llm_judge_score is not None]
    if len(both) >= 2:
        agree_exact = sum(1 for r in both if r.likert_score == r.llm_judge_score)
        agree_ge4 = sum(1 for r in both if (r.likert_score >= 4) == (r.llm_judge_score >= 4))
        print(f"\n--- Concordância humano × automático ({len(both)} amostras com as duas notas) ---")
        print(f"  Nota exatamente igual: {100 * agree_exact / len(both):.1f}%")
        print(f"  Concordância em '≥4 ou não': {100 * agree_ge4 / len(both):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="nº de pacientes a amostrar (default 20)")
    parser.add_argument("--seed", type=float, default=0.42, help="seed do setseed() do Postgres, -1..1 (default 0.42)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="URL da API local (default 127.0.0.1:8000)")
    parser.add_argument("--token", default=os.environ.get("FL_API_TOKEN"), help="Bearer token, se FL_AUTH_REQUIRED=true")
    parser.add_argument("--evaluator", default=os.environ.get("USER", "desconhecido"), help="nome de quem avalia")
    parser.add_argument("--generate", action="store_true",
                         help="gera amostras novas sem pausar pra nota humana (uso: automação/Makefile)")
    parser.add_argument("--score-pending", action="store_true",
                         help="revisa amostras já geradas com nota humana ainda pendente")
    parser.add_argument("--report", action="store_true", help="só imprime o resumo agregado")
    args = parser.parse_args()

    if args.report:
        run_report(args)
    elif args.score_pending:
        run_score_pending(args)
    elif args.generate:
        run_evaluation(args, pause_for_human=False)
    else:
        run_evaluation(args, pause_for_human=True)


if __name__ == "__main__":
    main()
