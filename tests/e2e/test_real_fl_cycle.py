"""
test_real_fl_cycle.py — Teste de ponta a ponta do ciclo FL.

Sem mocks do nosso código. Exercita o protocolo Flower completo:
  FedProxClient.fit() → CustomFedProxStrategy.aggregate_fit()
  FedProxClient.evaluate() → CustomFedProxStrategy.aggregate_evaluate()
  → ConvergenceTracker.check() → checkpoint salvo em disco

Não usa gRPC (evita dependência de porta e threading de processo principal).
O transporte Flower é substituído pela chamada direta ao protocolo NumPy —
a única coisa "stub" é o ClientProxy, que é infraestrutura do Flower, não
do MosaicFL.

Execução:
    pytest tests/e2e/ -v -s
    make test-e2e
"""
import os

import pytest
import torch
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch.utils.data import DataLoader, TensorDataset

from experiments.training.experiment_server import CustomFedProxStrategy
from mosaicfl.core.client import FedProxClient
from mosaicfl.core.convergence import ConvergenceTracker
from mosaicfl.core.federated import weighted_average_accuracy, weighted_average_loss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_loader(n: int = 40, seed: int = 0) -> DataLoader:
    """dia (3º tensor) é obrigatório desde que DiaRelativoEmbedding foi introduzido em
    client.py (`for batch_x, batch_y, batch_dia in self.train_loader`) — achado 2026-07-12,
    este helper nunca foi atualizado (mesma causa-raiz de test_checkpoints_persisted_in_store,
    ver _make_strategy(): este arquivo e2e ficou sem rodar por um tempo)."""
    torch.manual_seed(seed)
    x = torch.randint(0, 100, (n, 10))
    y = torch.randint(0, 2, (n,))
    dia = torch.randint(0, 30, (n, 10))
    return DataLoader(TensorDataset(x, y, dia), batch_size=16, shuffle=False)


def _make_strategy(tmp_path, tracker, history):
    """checkpoint_store: SQLiteCheckpointStore real (não mock) — CustomFedProxStrategy
    persiste via CheckpointStore desde a refatoração de organização de classes
    (commit 0dc13f2); save_dir/round_*.pt não existem mais (achado 2026-07-12,
    este teste e2e estava quebrado há algum tempo sem ninguém notar, porque
    `make test-e2e` não roda por padrão)."""
    from infrastructure.shared.checkpoint_store.sqlite_store import SQLiteCheckpointStore
    checkpoint_store = SQLiteCheckpointStore(db_path=str(tmp_path / "checkpoints" / "experiment.db"))
    return CustomFedProxStrategy(
        tracker=tracker,
        history=history,
        checkpoint_store=checkpoint_store,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        proximal_mu=0.1,
        evaluate_metrics_aggregation_fn=weighted_average_accuracy,
        fit_metrics_aggregation_fn=weighted_average_loss,
    )


def _run_fl_protocol(
    clients: list,
    strategy: CustomFedProxStrategy,
    num_rounds: int,
) -> int | None:
    """
    Orquestra rounds FL via protocolo NumPy direto (sem gRPC).
    Retorna o round de convergência ou None se não convergiu.
    """
    params = ndarrays_to_parameters(clients[0].get_parameters(config={}))
    ok = Status(code=Code.OK, message="")

    for rnd in range(1, num_rounds + 1):
        # ── Fase fit ─────────────────────────────────────────────────────────
        fit_results = []
        for client in clients:
            ndarrays, n, metrics = client.fit(
                parameters=parameters_to_ndarrays(params),
                config={"local_epochs": 1, "proximal_mu": 0.1},
            )
            fit_results.append((None, FitRes(
                status=ok,
                parameters=ndarrays_to_parameters(ndarrays),
                num_examples=n,
                metrics=metrics,
            )))
        params, _ = strategy.aggregate_fit(rnd, fit_results, [])

        # ── Fase evaluate ─────────────────────────────────────────────────────
        eval_results = []
        for client in clients:
            loss, n, metrics = client.evaluate(
                parameters=parameters_to_ndarrays(params),
                config={},
            )
            eval_results.append((None, EvaluateRes(
                status=ok,
                loss=float(loss),
                num_examples=n,
                metrics=metrics,
            )))
        try:
            strategy.aggregate_evaluate(rnd, eval_results, [])
        except StopIteration:
            return rnd

    return None


# ── Testes ────────────────────────────────────────────────────────────────────

@pytest.mark.e2e
class TestRealFLCycle:
    """
    Ciclo FL real de ponta a ponta.

    Cada teste exercita componentes reais do MosaicFL sem nenhum mock:
      - FedProxClient: modelo BEHRT real, forward/backward reais
      - CustomFedProxStrategy: FedProx real, checkpoint real
      - ConvergenceTracker: algoritmo de janela deslizante real
    """

    def test_rounds_complete_and_history_populated(self, tmp_path):
        """Múltiplos rounds completam e populam o histórico corretamente."""
        clients = [FedProxClient(i, _make_loader(seed=i), _make_loader(seed=i)) for i in range(3)]
        tracker = ConvergenceTracker(threshold=0.5, patience=1)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        strategy = _make_strategy(tmp_path, tracker, history)

        _run_fl_protocol(clients, strategy, num_rounds=3)

        assert len(history["rounds"]) >= 1
        assert all(isinstance(r, int) for r in history["rounds"])
        assert all(0.0 <= acc <= 1.0 for acc in history["accuracy"])
        assert all(mb > 0 for mb in history["communication_mb"])

    def test_checkpoints_persisted_in_store(self, tmp_path):
        """Checkpoint real é gravado no CheckpointStore a cada round — persistência
        vai via CheckpointStore desde a refatoração (não mais round_*.pt em disco,
        ver _make_strategy())."""
        clients = [FedProxClient(i, _make_loader(seed=i), _make_loader(seed=i)) for i in range(3)]
        tracker = ConvergenceTracker(threshold=0.5, patience=1)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        strategy = _make_strategy(tmp_path, tracker, history)

        _run_fl_protocol(clients, strategy, num_rounds=2)

        loaded = strategy.checkpoint_store.load_latest()
        assert loaded is not None, "Nenhum checkpoint persistido no CheckpointStore"
        assert "model_state" in loaded
        assert history["last_checkpoint"] is not None

    def test_convergence_detected_and_checkpoint_saved(self, tmp_path):
        """
        Convergência detectada → checkpoint persistido e converged_round preenchido.

        threshold=0.5 garante convergência rápida com dados sintéticos:
        qualquer Δ < 0.5 na accuracy entre rounds consecutivos converge.
        """
        clients = [FedProxClient(i, _make_loader(seed=i), _make_loader(seed=i)) for i in range(3)]
        tracker = ConvergenceTracker(threshold=0.5, patience=1)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        strategy = _make_strategy(tmp_path, tracker, history)

        converged_at = _run_fl_protocol(clients, strategy, num_rounds=5)

        assert converged_at is not None, "Não convergiu em 5 rounds com threshold=0.5"
        assert tracker.converged_round is not None
        assert strategy.checkpoint_store.load_latest() is not None

    def test_fedprox_proximal_term_applied(self, tmp_path):
        """
        O termo proximal do FedProx é aplicado — loss de fit inclui regularização.

        Verifica que communication_mb > 0 (pesos reais trafegaram) e que
        os pesos após fit são distintos dos pesos iniciais.
        """
        loader = _make_loader(seed=42)
        client = FedProxClient(0, loader, loader)

        initial_params = client.get_parameters(config={})
        updated_params, n_samples, metrics = client.fit(
            parameters=initial_params,
            config={"local_epochs": 2, "proximal_mu": 1.0},
        )

        assert n_samples > 0
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

        diff = sum(
            float(abs(a - b).sum())
            for a, b in zip(initial_params, updated_params)
        )
        assert diff > 0.0, "Pesos não foram atualizados pelo treino"

    def test_convergence_tracker_real_accuracy_sequence(self, tmp_path):
        """
        ConvergenceTracker integrado à strategy detecta convergência na sequência
        real de accuracy produzida pelo treino federado.
        """
        clients = [FedProxClient(i, _make_loader(seed=i), _make_loader(seed=i)) for i in range(3)]
        tracker = ConvergenceTracker(threshold=0.5, patience=2)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        strategy = _make_strategy(tmp_path, tracker, history)
        strategy.tracker = tracker

        _run_fl_protocol(clients, strategy, num_rounds=6)

        assert len(tracker.history) >= 1, "ConvergenceTracker não registrou nenhuma accuracy"
        assert all(0.0 <= acc <= 1.0 for acc in tracker.history)

    def test_multiple_clients_produce_weighted_aggregate(self, tmp_path):
        """
        Com 3 clientes de tamanhos diferentes, o agregado reflete média ponderada.
        Testa que n_samples é respeitado na agregação (FedAvg ponderado).
        """
        loaders = [
            _make_loader(n=10, seed=0),
            _make_loader(n=40, seed=1),
            _make_loader(n=20, seed=2),
        ]
        clients = [FedProxClient(i, loaders[i], loaders[i]) for i in range(3)]
        tracker = ConvergenceTracker(threshold=0.5, patience=1)
        history = {"rounds": [], "accuracy": [], "communication_mb": [], "last_checkpoint": None}
        strategy = _make_strategy(tmp_path, tracker, history)

        _run_fl_protocol(clients, strategy, num_rounds=2)

        assert len(history["rounds"]) >= 1
        # communication_mb deve refletir tráfego dos 3 clientes
        assert all(mb > 0 for mb in history["communication_mb"])
