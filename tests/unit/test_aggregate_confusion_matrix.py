"""
Testes para a agregação de confusion_matrix_json em
weighted_average_evaluate_metrics() — achado 2026-07-29. Soma célula a célula
entre hospitais (equivalente matemático de computar uma matriz global), NÃO
média ponderada como accuracy/F1.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import weighted_average_evaluate_metrics


class TestAggregateConfusionMatrix:
    def test_sums_matrices_cell_by_cell(self):
        cm_bpsp = [[10, 2], [1, 8]]
        cm_hsl  = [[3, 0], [1, 5]]
        metrics = [
            (21, {"accuracy": 0.8, "f1_macro": 0.5, "confusion_matrix_json": json.dumps(cm_bpsp)}),
            (9,  {"accuracy": 0.7, "f1_macro": 0.4, "confusion_matrix_json": json.dumps(cm_hsl)}),
        ]
        result = weighted_average_evaluate_metrics(metrics)
        summed = json.loads(result["confusion_matrix_json"])
        assert summed == [[13, 2], [2, 13]]

    def test_absent_when_no_client_sends(self):
        metrics = [(100, {"accuracy": 0.8, "f1_macro": 0.5})]
        result = weighted_average_evaluate_metrics(metrics)
        assert "confusion_matrix_json" not in result

    def test_single_client_returns_its_own_matrix(self):
        cm = [[5, 1], [0, 4]]
        metrics = [(10, {"accuracy": 0.9, "f1_macro": 0.85, "confusion_matrix_json": json.dumps(cm)})]
        result = weighted_average_evaluate_metrics(metrics)
        assert json.loads(result["confusion_matrix_json"]) == cm
