"""models.py — Dataclasses de termos pendentes e resultado de validação."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from integration.column_resolver import looks_like_valid_analyte_name

logger = logging.getLogger(__name__)


@dataclass
class PendingTerm:
    alias: str
    canonical_proposto: str
    source: str
    term_type: str

    def __str__(self) -> str:
        return (
            f"  alias={self.alias!r:45s}  "
            f"canonical_proposto={self.canonical_proposto!r:30s}  "
            f"source={self.source!r}"
        )


@dataclass
class ValidationResult:
    total_analitos: int
    resolvidos: int
    pendentes: list[PendingTerm] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.pendentes) == 0

    @property
    def suspeitos(self) -> list[PendingTerm]:
        """Pendentes cujo canonical proposto não parece um nome de exame real
        (hoje: puramente numérico) — achado 2026-07-26, canonical="183" (código
        interno de laboratório do HSL nunca traduzido, ver
        integration/column_resolver.py::looks_like_valid_analyte_name). Esses
        passam batido pelo critério de "alias verdadeiro" de
        activate_all_auto_normalized() porque normalize("183").upper() == "183"
        — alias bruto e canonical proposto são idênticos, igual a uma variante
        de grafia legítima."""
        return [p for p in self.pendentes if not looks_like_valid_analyte_name(p.canonical_proposto)]

    def print_report(self) -> None:
        print(f"\nValidação de analitos: {self.resolvidos}/{self.total_analitos} resolvidos")
        if self.pendentes:
            print(f"\n{'─'*100}")
            print(f"{'ALIAS ORIGINAL':<45}  {'CANONICAL PROPOSTO':<30}  FONTE")
            print(f"{'─'*100}")
            for p in sorted(self.pendentes, key=lambda x: x.canonical_proposto):
                print(p)
            print(f"{'─'*100}")
            print(
                f"\n⚠ {len(self.pendentes)} termo(s) com active=FALSE. "
                "Corrija ou ative antes de prosseguir com a carga.\n"
                "  → list_pending_terms()           para ver todos os pendentes\n"
                "  → activate_term(alias, ...)      para ativar o canonical proposto\n"
                "  → correct_term(alias, novo, ...) para corrigir o canonical e ativar\n"
            )
            if self.suspeitos:
                nomes = ", ".join(p.canonical_proposto for p in self.suspeitos)
                print(
                    f"\n🚩 {len(self.suspeitos)} termo(s) com canonical proposto suspeito "
                    f"(não parece nome de exame real — provável código bruto não traduzido "
                    f"na fonte): {nomes}\n"
                    "  → NÃO use activate_all_auto_normalized() sem revisar estes antes.\n"
                    "  → correct_term(alias, nome_correto, conn) pra cada um.\n"
                )
                logger.warning(
                    "scan_analytes_suspicious_canonicals count=%d nomes=%s — "
                    "revisar antes de ativar (não parecem nomes de exame reais)",
                    len(self.suspeitos), nomes,
                )
        else:
            print("✓ Todos os analitos têm canonical ativo. Carga liberada.\n")
