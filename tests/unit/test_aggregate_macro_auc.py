"""
Testes para a agregação de macro_auc em weighted_average_evaluate_metrics() —
achado 2026-07-26: fl_trainings.macro_auc ficava sempre NULL no Caminho B, AUC-ROC
nunca era calculado em lugar nenhum (F1 já era). Client.py agora calcula AUC-ROC
macro (one-vs-rest) localmente (ver _macro_auc_ovr), omitindo a chave quando nenhuma
classe tem as duas categorias presentes — a agregação precisa lidar com isso sem
quebrar quando só parte dos clientes envia.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import weighted_average_evaluate_metrics


class TestAggregateMacroAuc:
    def test_weighted_average_when_all_clients_send(self):
        metrics = [
            (100, {"accuracy": 0.8, "f1_macro": 0.5, "macro_auc": 0.9}),
            (300, {"accuracy": 0.7, "f1_macro": 0.4, "macro_auc": 0.7}),
        ]
        result = weighted_average_evaluate_metrics(metrics)
        expected = (100 * 0.9 + 300 * 0.7) / 400
        assert abs(result["macro_auc"] - expected) < 1e-6

    def test_absent_when_no_client_sends(self):
        metrics = [(100, {"accuracy": 0.8, "f1_macro": 0.5})]
        result = weighted_average_evaluate_metrics(metrics)
        assert "macro_auc" not in result

    def test_averages_only_over_clients_that_sent(self):
        """1 cliente sem macro_auc (ex.: todas as classes degeneradas localmente) —
        não deve puxar a média pra baixo com um 0.0 implícito, nem quebrar."""
        metrics = [
            (100, {"accuracy": 0.8, "f1_macro": 0.5, "macro_auc": 0.9}),
            (300, {"accuracy": 0.7, "f1_macro": 0.4}),  # sem macro_auc
        ]
        result = weighted_average_evaluate_metrics(metrics)
        assert result["macro_auc"] == 0.9
        # accuracy/f1_macro continuam agregados normalmente com os 2 clientes
        expected_acc = (100 * 0.8 + 300 * 0.7) / 400
        assert abs(result["accuracy"] - expected_acc) < 1e-6

    def test_single_client(self):
        metrics = [(50, {"accuracy": 0.8, "f1_macro": 0.5, "macro_auc": 0.85})]
        result = weighted_average_evaluate_metrics(metrics)
        assert abs(result["macro_auc"] - 0.85) < 1e-6
