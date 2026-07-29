"""confusion_stats.py — Estatísticas derivadas de uma matriz de confusão
agregada (precisão, recall/sensibilidade, especificidade por classe, e
intervalo de confiança de Wilson para a acurácia como proporção p̂), mais
teste de hipótese de duas proporções (p' agrupado) pra comparar a acurácia
de dois treinos/configurações — achado 2026-07-30, confirmado com a autora
(curso de estatística, mesma professora que orienta o colega que pediu essa
métrica): p̂ = proporção amostral (accuracy de UM treino); p' = proporção
AGRUPADA ("pooled"), usada como estimador de H0 (as duas amostras vêm da
mesma proporção populacional) num teste-z de duas proporções — devolve um
p-valor de verdade pra "essa diferença de acurácia é estatisticamente
significativa?", não só a diferença bruta em pontos percentuais.

Achado 2026-07-29: matriz de confusão, precisão e recall por classe já
existiam no Caminho A (src/mosaicfl/core/evaluation.py::evaluate(), test_loader
centralizado), mas nunca no Caminho B em produção (só via _run_calibration,
código morto por exigir o mesmo test_loader centralizado, indisponível por
design de privacidade). Este módulo consome a matriz de confusão JÁ agregada
pelo servidor (soma célula a célula entre hospitais, ver
mosaicfl.core.federated::weighted_average_evaluate_metrics) — nunca opera
sobre predição/rótulo bruto por paciente.
"""
import math
from typing import Dict, List, Optional, Tuple


def wilson_score_interval(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de confiança de Wilson (Wilson, 1927) pra uma proporção p̂
    estimada de n observações — mais preciso que a aproximação normal simples
    quando p̂ está perto de 0/1 ou n é pequeno, e não depende de scipy.
    z=1.96 → 95% de confiança (default)."""
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z ** 2 / n
    center = p_hat + z ** 2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def derive_stats_from_confusion_matrix(
    cm: List[List[int]], class_labels: List[str],
) -> Dict:
    """cm[i][j] = nº de casos com classe REAL i previstos como classe j.

    Retorna precisão/recall(sensibilidade)/especificidade/suporte por classe
    (None quando o denominador é zero — nenhuma amostra daquele tipo na
    matriz agregada, não confundir com 0.0), mais accuracy (p̂) com intervalo
    de confiança de Wilson 95%.
    """
    n_classes = len(cm)
    total = sum(sum(row) for row in cm)

    per_class: Dict[str, Dict] = {}
    for i, label in enumerate(class_labels[:n_classes]):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n_classes)) - tp
        tn = total - tp - fn - fp
        support = sum(cm[i])
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
        specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else None
        per_class[label] = {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "support": support,
        }

    accuracy = sum(cm[i][i] for i in range(n_classes)) / total if total > 0 else 0.0
    ci_lo, ci_hi = wilson_score_interval(accuracy, total)

    return {
        "confusion_matrix": cm,
        "class_labels": list(class_labels[:n_classes]),
        "per_class_stats": per_class,
        "accuracy_p_hat": round(accuracy, 4),
        "accuracy_ci_95_wilson": [round(ci_lo, 4), round(ci_hi, 4)],
        "n_total": total,
    }


def _standard_normal_cdf(z: float) -> float:
    """Φ(z) — CDF da normal padrão via math.erf (stdlib), sem depender de
    scipy. Fórmula exata: Φ(z) = 0.5·(1 + erf(z/√2))."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def pooled_proportion(p_hat_1: float, n1: int, p_hat_2: float, n2: int) -> float:
    """p' — proporção agrupada de duas amostras, estimador de H0 (as duas
    vêm da mesma proporção populacional). p' = (x1+x2)/(n1+n2), onde
    x_i = p̂_i · n_i (nº de acertos)."""
    if n1 + n2 == 0:
        return 0.0
    x1 = p_hat_1 * n1
    x2 = p_hat_2 * n2
    return (x1 + x2) / (n1 + n2)


def two_proportion_z_test(
    p_hat_1: float, n1: int, p_hat_2: float, n2: int, alpha: float = 0.05,
) -> Dict:
    """Teste-z de duas proporções (comparação de duas acurácias, ex.: treino
    A vs. treino B) usando a proporção agrupada p' pra estimar o erro padrão
    sob H0 (p1=p2). Bicaudal — não presume de antemão qual dos dois é melhor.

    Retorna None nos campos estatísticos quando n1 ou n2 é 0 (não dá pra
    testar) — sempre com os p̂/n brutos disponíveis pra quem for revisar.
    """
    result: Dict = {
        "p_hat_1": round(p_hat_1, 4), "n1": n1,
        "p_hat_2": round(p_hat_2, 4), "n2": n2,
        "diff": round(p_hat_1 - p_hat_2, 4),
        "p_pooled": None, "z_statistic": None, "p_value": None,
        "alpha": alpha, "significant": None,
    }
    if n1 <= 0 or n2 <= 0:
        return result

    p_pooled = pooled_proportion(p_hat_1, n1, p_hat_2, n2)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    z = (p_hat_1 - p_hat_2) / se if se > 0 else 0.0
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))

    result.update({
        "p_pooled": round(p_pooled, 4),
        "z_statistic": round(z, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
    })
    return result
