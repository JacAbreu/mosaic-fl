"""models.py — Dataclasses de termos pendentes e resultado de validação."""
from __future__ import annotations

from dataclasses import dataclass, field


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
        else:
            print("✓ Todos os analitos têm canonical ativo. Carga liberada.\n")
