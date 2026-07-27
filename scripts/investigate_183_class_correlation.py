"""
investigate_183_class_correlation.py — Verifica se os atendimentos que tinham
exames de Amilase vindos de UTI/Internação/Pronto Socorro (proxy dos registros
que antes carregavam o canonical bugado "183", corrigido pela migration 026)
se concentram desproporcionalmente nas classes de prognóstico que colapsaram
(curado_internado, melhora_pronto) — achado 2026-07-26/27, ver
docs/Linha_do_Tempo_MOSAIC-FL.md.

Depois da migration 026, exam_records.analyte='183' virou 'AMILASE' em todo
lugar — não sobrou marca direta de "era 183 antes". Usa origin (DE_ORIGEM
original) como proxy: no CSV bruto do HSL, 92,6% dos registros "183" vinham de
UTI/Unidades de Internação/Pronto Socorro (achado ao vivo, ver conversa da
sessão) — bem diferente do padrão normal de pedido de Amilase (LAB/HOSP no
BPSP). Não é uma marca perfeita (pode incluir alguns AMILASE legítimos desses
mesmos departamentos, e pode perder alguns "183" de origem incomum, tipo
Ecocardiografia/Holter), mas é o melhor proxy disponível sem um backup dos
dados brutos pré-migration.

Compara a distribuição das 5 classes de prognóstico (mesma lógica de
src/mosaicfl/core/preprocessor/outcomes.py::_map_outcome, reaproveitada aqui
sem duplicar) entre:
  - Grupo A: atendimentos com AMILASE de origem UTI/Internação/Pronto Socorro
  - Grupo B: todos os atendimentos (linha de base)

Uso:
    FL_DB_URL=postgresql://mosaicfl:senha@localhost:PORTA/BANCO \\
        python3 scripts/investigate_183_class_correlation.py
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mosaicfl.core.preprocessor.outcomes import _map_outcome

_CLASS_NAMES = {
    0: "curado_pronto",
    1: "curado_internado",
    2: "melhora_pronto",
    3: "melhora_internado_breve",
    4: "melhora_internado_grave",
}

_HIGH_ACUITY_ORIGINS = ("UTI", "Unidades de Internação", "Pronto Socorro")

_SQL = """
SELECT DISTINCT a.attendance_id, a.attendance_type, co.outcome_class,
       (co.outcome_at - a.attended_at) AS duration_days,
       EXISTS (
           SELECT 1 FROM metrics.exam_records e2
           WHERE e2.attendance_id = a.attendance_id
             AND e2.analyte = 'AMILASE'
             AND e2.origin = ANY(:origins)
       ) AS teve_amilase_alta_gravidade
FROM clinical.attendances a
JOIN metrics.clinical_outcomes co ON co.attendance_id = a.attendance_id
WHERE co.outcome_class NOT IN (2, 3, 4)
  AND (co.outcome_at - a.attended_at) >= 0
"""


def _distribution(rows) -> Counter:
    counts: Counter = Counter()
    for r in rows:
        cls = _map_outcome(r.outcome_class, float(r.duration_days), r.attendance_type)
        if cls != -1:
            counts[cls] += 1
    return counts


def _print_distribution(label: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n{label} (n={total})")
    if total == 0:
        print("  (sem atendimentos)")
        return
    for cls in range(5):
        n = counts.get(cls, 0)
        pct = 100 * n / total
        print(f"  {_CLASS_NAMES[cls]:<28} {n:>6}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=os.getenv("FL_DB_URL"))
    args = parser.parse_args()
    if not args.db_url:
        print("ERRO: defina FL_DB_URL ou passe --db-url", file=sys.stderr)
        sys.exit(1)

    import sqlalchemy as sa
    engine = sa.create_engine(args.db_url)
    with engine.connect() as conn:
        rows = conn.execute(sa.text(_SQL), {"origins": list(_HIGH_ACUITY_ORIGINS)}).fetchall()

    grupo_a = [r for r in rows if r.teve_amilase_alta_gravidade]
    grupo_b = rows  # linha de base: todos os atendimentos

    print("=" * 70)
    print("Correlação entre AMILASE de origem UTI/Internação/Pronto Socorro")
    print("(proxy do antigo canonical '183') e classe de prognóstico")
    print("=" * 70)
    _print_distribution(
        f"Grupo A — AMILASE de {'/'.join(_HIGH_ACUITY_ORIGINS)}",
        _distribution(grupo_a),
    )
    _print_distribution("Grupo B — todos os atendimentos (linha de base)", _distribution(grupo_b))
    print()


if __name__ == "__main__":
    main()
