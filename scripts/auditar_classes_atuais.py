"""
auditar_classes_atuais.py — Reproduz a distribuição das 5 classes de prognóstico
atuais (curado_pronto, curado_internado, melhora_pronto, melhora_internado_breve,
melhora_internado_grave) e audita o limiar de 10 dias usado em _map_outcome().

Não é um script de "descoberta" de classes (as classes atuais vieram de uma decisão
de desenho clínico, não de exploração de dados — ver docs/Racional_Classes_Prognostico.md).
É uma ferramenta de auditoria: mostra o que a regra atual produz, e se o corte de 10
dias é um bom separador estatístico ou um valor arbitrário.

Uso:
    export FL_DB_URL="postgresql://user:pass@localhost:PORTA/BANCO"
    python scripts/auditar_classes_atuais.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from sqlalchemy import create_engine, text

from mosaicfl.core.preprocessor import _map_outcome

_CLASS_NAMES = {
    0: "curado_pronto",
    1: "curado_internado",
    2: "melhora_pronto",
    3: "melhora_internado_breve",
    4: "melhora_internado_grave",
}

_QUERY = """
SELECT
    a.hospital_id,
    a.attendance_type,
    co.outcome_class,
    (co.outcome_at - a.attended_at) AS duration_days
FROM clinical.attendances       a
JOIN metrics.clinical_outcomes  co ON co.attendance_id = a.attendance_id
WHERE co.outcome_class NOT IN (2, 3, 4)
  AND (co.outcome_at - a.attended_at) >= 0
"""


def main() -> None:
    db_url = os.environ.get("FL_DB_URL")
    if not db_url:
        print("ERRO: defina FL_DB_URL antes de rodar.")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql(text(_QUERY), conn)

    print(f"Atendimentos elegíveis (outcome 0/1, datas consistentes): {len(df):,}\n")

    # Aplica a MESMA função usada em produção — garante que a auditoria reflete
    # exatamente o que o pipeline real produz, não uma reimplementação paralela.
    df["classe"] = df.apply(
        lambda r: _map_outcome(r["outcome_class"], r["duration_days"], r["attendance_type"]),
        axis=1,
    )
    df["classe_nome"] = df["classe"].map(_CLASS_NAMES)

    print("=" * 60)
    print("Distribuição das 5 classes — geral")
    print("=" * 60)
    geral = df["classe_nome"].value_counts()
    geral_pct = (geral / len(df) * 100).round(1)
    for nome in _CLASS_NAMES.values():
        n = geral.get(nome, 0)
        pct = geral_pct.get(nome, 0.0)
        print(f"  {nome:<26} {n:>8,}  ({pct:>5.1f}%)")

    print("\n" + "=" * 60)
    print("Distribuição por hospital (%) — heterogeneidade non-IID")
    print("=" * 60)
    tabela = pd.crosstab(df["hospital_id"], df["classe_nome"], normalize="index") * 100
    print(tabela.round(1).to_string())

    # --- Auditoria do limiar de 10 dias -------------------------------------
    print("\n" + "=" * 60)
    print("Auditoria do limiar de 10 dias (melhora_internado_breve vs. grave)")
    print("=" * 60)
    internados_melhora = df[(df["outcome_class"] == 1) & (df["attendance_type"] == "Internado")]
    if len(internados_melhora) == 0:
        print("  Nenhum registro internado com outcome=melhora encontrado neste banco.")
        return

    desc = internados_melhora["duration_days"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(desc.to_string())

    n_breve = (internados_melhora["duration_days"] <= 10).sum()
    n_grave = (internados_melhora["duration_days"] > 10).sum()
    total = len(internados_melhora)
    print(f"\n  Com o corte atual (10 dias): breve={n_breve:,} ({n_breve/total*100:.1f}%), "
          f"grave={n_grave:,} ({n_grave/total*100:.1f}%)")

    mediana = internados_melhora["duration_days"].median()
    print(f"  Mediana real de duration_days neste grupo: {mediana:.1f} dias")
    if abs(mediana - 10) > 3:
        print(f"  >>> O corte de 10 dias está longe da mediana ({mediana:.1f}) — "
              f"pode estar produzindo uma divisão desbalanceada entre breve/grave.")
    else:
        print("  >>> O corte de 10 dias está próximo da mediana — divisão razoavelmente equilibrada.")


if __name__ == "__main__":
    main()
