"""
outcomes.py — Mapeamento de desfecho clínico para as 5 classes de prognóstico do BEHRT.
"""

# outcome_class vem de classify_outcome() em integration/fapesp/transforms.py.
# O mapeamento para 5 classes é feito por _map_outcome() combinando outcome_class,
# duration_days e attendance_type. Classes 2, 3, 4 são excluídas na query SQL.


def _map_outcome(
    outcome_class: int,
    duration_days: float,
    attendance_type: str,
    internado_breve_max_days: int = 10,
) -> int:
    """
    Converte (outcome_class, duration_days, attendance_type) em classe de prognóstico.

    5 classes que cruzam desfecho clínico, tipo de atendimento e duração:

      0 = curado_pronto           — outcome 0, não-internado (ambulatorial/pronto/externo)
      1 = curado_internado        — outcome 0, internado (curso grave com recuperação completa)
      2 = melhora_pronto          — outcome 1, não-internado (COVID moderado, melhora sem internação)
      3 = melhora_internado_breve — outcome 1, internado, ≤ internado_breve_max_days
      4 = melhora_internado_grave — outcome 1, internado, > internado_breve_max_days

    O tipo de atendimento define a trajetória clínica: ambulatorial/pronto = acesso pontual;
    internado = internação contínua com acompanhamento diário de exames.

    internado_breve_max_days: limiar em dias entre classes 3 e 4 — decisão de desenho
    clínico (commit 6193d45), nunca validada estatisticamente. Achado 2026-07-06: mediana
    real de duration_days é 5, não 10 (ver docs/Racional_Classes_Prognostico.md). Default
    10 preserva o comportamento histórico; produção deve passar MODEL_CFG.internado_breve_max_days
    (src/mosaicfl/core/config.py) em vez de confiar neste default.

    Retorna -1 para outcome_class não mapeado (dado censurado ou excluído da análise).
    """
    internado = str(attendance_type).strip() == "Internado"

    if outcome_class == 0:
        return 1 if internado else 0
    if outcome_class == 1:
        if not internado:
            return 2
        return 3 if duration_days <= internado_breve_max_days else 4
    return -1  # 4=censored, 5=uti, 6=obito — excluídos ou ausentes nos dados FAPESP
