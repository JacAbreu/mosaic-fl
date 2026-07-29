"""
Testes para a agregação de rag_precision_at_k em weighted_average_evaluate_metrics()
— achado 2026-07-28: Precision@k do RAG só existia no Caminho A (test_loader
centralizado). Portado pro Caminho B com avaliação client-side (ver
mosaicfl.core.rag.precision.eval_precision_at_k, mosaicfl.core.client.FedProxClient.
evaluate()) — mesma filosofia de agregação server-side do macro_auc: só entre quem
enviou, nunca puxa a média com um 0.0 implícito de quem não avaliou.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import weighted_average_evaluate_metrics


class TestAggregateRagPrecisionAtK:
    def test_weighted_average_when_all_clients_send(self):
        metrics = [
            (100, {"accuracy": 0.8, "f1_macro": 0.5, "rag_precision_at_k": 0.9, "rag_k": 3}),
            (300, {"accuracy": 0.7, "f1_macro": 0.4, "rag_precision_at_k": 0.6, "rag_k": 3}),
        ]
        result = weighted_average_evaluate_metrics(metrics)
        expected = (100 * 0.9 + 300 * 0.6) / 400
        assert abs(result["rag_precision_at_k"] - expected) < 1e-6
        assert result["rag_k"] == 3

    def test_absent_when_no_client_sends(self):
        """1ª rodada de avaliação de cada treino — servidor ainda não construiu
        knowledge base nenhuma, nenhum cliente recebeu rag_patterns_json."""
        metrics = [(100, {"accuracy": 0.8, "f1_macro": 0.5})]
        result = weighted_average_evaluate_metrics(metrics)
        assert "rag_precision_at_k" not in result
        assert "rag_k" not in result

    def test_averages_only_over_clients_that_sent(self):
        metrics = [
            (100, {"accuracy": 0.8, "f1_macro": 0.5, "rag_precision_at_k": 0.9, "rag_k": 3}),
            (300, {"accuracy": 0.7, "f1_macro": 0.4}),  # sem rag_precision_at_k
        ]
        result = weighted_average_evaluate_metrics(metrics)
        assert result["rag_precision_at_k"] == 0.9

    def test_single_client(self):
        metrics = [(50, {"accuracy": 0.8, "f1_macro": 0.5, "rag_precision_at_k": 0.75, "rag_k": 3})]
        result = weighted_average_evaluate_metrics(metrics)
        assert abs(result["rag_precision_at_k"] - 0.75) < 1e-6
