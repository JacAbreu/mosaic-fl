"""
set_class_weight_overrides.py — Grava peso de classe explícito (cost-sensitive
learning, Strategy pattern em mosaicfl.core.class_weighting) em
clinical.fl_orchestration_config, no banco LOCAL desta máquina.

Como BPSP e HSL têm bancos separados (dado clínico nunca sai do hospital), cada
hospital tem sua PRÓPRIA linha dessa tabela — rodar este script no desktop (BPSP,
via `make server-set-class-weights`) e no notebook (HSL, via
`make client-set-class-weights`) grava valores INDEPENDENTES em cada máquina.
Isso é intencional: dá pra ter pesos diferentes por hospital (ex.: melhora_pronto
é maioria no HSL e rara no BPSP — um peso único não serve pros dois, ver
docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção 14).

FedProxClient lê essa mesma tabela LOCAL como 2º nível de prioridade (depois do
canal compartilhado servidor->cliente de clinical.fl_orchestration_config no
banco do SERVIDOR, que continua existindo pra quem quiser forçar o MESMO peso
nos dois hospitais — ver docs/Linha_do_Tempo_MOSAIC-FL.md).

Uso:
    FL_DB_URL=postgresql://mosaicfl:senha@localhost:PORTA/BANCO \\
        python3 scripts/set_class_weight_overrides.py --overrides '{"curado_internado": 25}'

    # Limpar (volta tudo pra class_balanced nesta máquina):
    FL_DB_URL=... python3 scripts/set_class_weight_overrides.py --clear

    # Ver o que está gravado hoje, sem alterar nada:
    FL_DB_URL=... python3 scripts/set_class_weight_overrides.py --show
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mosaicfl.core.class_weighting import validate_overrides
from mosaicfl.core.config import MODEL_CFG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-url", default=os.getenv("FL_DB_URL"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overrides", help='JSON {"classe": peso, ...} — substitui o conjunto inteiro')
    group.add_argument("--clear", action="store_true", help="remove todos os overrides desta máquina")
    group.add_argument("--show", action="store_true", help="só mostra o valor atual, não grava nada")
    args = parser.parse_args()

    if not args.db_url:
        print("ERRO: defina FL_DB_URL ou passe --db-url", file=sys.stderr)
        sys.exit(1)

    import sqlalchemy as sa
    engine = sa.create_engine(args.db_url)

    if args.show:
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT class_weight_overrides_json FROM clinical.fl_orchestration_config WHERE id = 'current'"
            )).mappings().first()
        current = (row["class_weight_overrides_json"] if row else None) or {}
        print(json.dumps(current, indent=2, ensure_ascii=False))
        return

    overrides = {} if args.clear else json.loads(args.overrides)
    try:
        validate_overrides(overrides, MODEL_CFG.class_labels)
    except ValueError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(1)

    overrides_json = json.dumps(overrides) if overrides else None
    with engine.begin() as conn:
        conn.execute(sa.text("""
            UPDATE clinical.fl_orchestration_config
            SET class_weight_overrides_json = cast(:overrides AS jsonb), updated_at = now()
            WHERE id = 'current'
        """), {"overrides": overrides_json})

    if overrides:
        print(f"Gravado nesta máquina: {json.dumps(overrides, ensure_ascii=False)}")
        untouched = [c for c in MODEL_CFG.class_labels if c not in overrides]
        if untouched:
            print(f"Sem override (class_balanced): {untouched}")
    else:
        print("Overrides removidos — todas as classes voltam pra class_balanced nesta máquina.")


if __name__ == "__main__":
    main()
