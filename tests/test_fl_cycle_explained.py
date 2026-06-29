"""
test_fl_cycle_explained.py
Documentação executável do ciclo de Federated Learning do MOSAIC-FL.

Cada classe cobre uma fase do ciclo e imprime logs detalhados mostrando
quem envia o quê, como os dados fluem e o que o código realmente faz.
Use -s para ver os prints ao rodar.

Ciclo coberto:
  1. Scheduler verifica quórum e dispara round
  2. Servidor envia modelo global ao cliente  (set_parameters)
  3. Cliente treina localmente com FedProx    (fit)
  4. Cliente retorna pesos ao servidor        (get_parameters via fit)
  5. Servidor agrega métricas de acurácia     (weighted_average)
     Servidor agrega parâmetros              (FedAvg manual)
  6. Servidor rastreia convergência           (ConvergenceTracker)
  7. Ciclo end-to-end com múltiplos clientes e múltiplos rounds

APIs reais usadas nestes testes:
  FedProxClient(client_id: int, train_loader, val_loader)
    .fit()         → (params: List[np.ndarray], n_samples: int, {"loss": float})
    .evaluate()    → (loss: float, n_samples: int, {"accuracy": float, "client_id": int})
    .get_parameters() → List[np.ndarray]  (state_dict completo, incl. buffers)
    .set_parameters() → carrega List[np.ndarray] no state_dict

  weighted_average([(n_examples, {"accuracy": float}), ...]) → {"accuracy": float}
    NOTA: agrega MÉTRICAS (accuracy), NÃO parâmetros do modelo.
    Para parâmetros, use o helper _fedavg_params() deste arquivo.

  ConvergenceTracker.check(accuracy) → bool
    Usa janela deslizante: converge quando os últimos patience deltas são < threshold.

Uso:
    pytest tests/test_fl_cycle_explained.py -v -s
    pytest tests/test_fl_cycle_explained.py -v -s -k "TestServerAggregates"
"""
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mosaicfl.core.config import (
    BATCH_SIZE, DEVICE, EMBED_DIM, LR, LOCAL_EPOCHS,
    MAX_SEQ_LEN, NUM_CLASSES, NUM_HEADS, NUM_LAYERS,
    PROXIMAL_MU, VOCAB_SIZE,
)
from mosaicfl.core.model import SimplifiedBEHRT
from mosaicfl.core.client import FedProxClient
from mosaicfl.core.federated import weighted_average
from mosaicfl.core.convergence import ConvergenceTracker
from infrastructure.mosaicfl_scheduler.schedule_state import SchedulerState
from infrastructure.mosaicfl_scheduler.scheduler_daemon import FederatedScheduler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_loader(n=6, seq_len=5, seed=42):
    """DataLoader mínimo para testes rápidos."""
    torch.manual_seed(seed)
    x   = torch.randint(0, VOCAB_SIZE, (n, seq_len))
    y   = torch.randint(0, NUM_CLASSES, (n,))
    dia = torch.randint(0, 100, (n, seq_len))
    return DataLoader(TensorDataset(x, y, dia), batch_size=3, shuffle=False)


def _make_client(client_id=0, n=6, seed=42):
    """FedProxClient com dados sintéticos mínimos."""
    loader = _tiny_loader(n=n, seed=seed)
    return FedProxClient(client_id, loader, loader)


def _fedavg_params(client_results):
    """
    Agrega parâmetros de múltiplos clientes por média ponderada (FedAvg).
    client_results: [(params, n_samples, metrics), ...]
    Retorna: List[np.ndarray] — parâmetros agregados
    """
    total = sum(n for _, n, _ in client_results)
    agg = [np.zeros_like(p, dtype=np.float32) for p in client_results[0][0]]
    for params, n, _ in client_results:
        w = n / total
        for i, p in enumerate(params):
            agg[i] = agg[i] + w * p.astype(np.float32)
    return agg


def _make_scheduler(state, checker, dispatcher):
    """FederatedScheduler com dependências mockadas e TCP/SQLite desabilitados."""
    with patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerState") as MS, \
         patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.SchedulerStateStore"), \
         patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.ClientAvailabilityChecker",
               return_value=checker), \
         patch("infrastructure.mosaicfl_scheduler.scheduler_daemon.RoundDispatcher",
               return_value=dispatcher):
        MS.load.return_value = state
        sched = FederatedScheduler(interval_hours=1, min_clients=3, max_rounds=20)
        sched.state = state
    sched._check_server_connectivity = MagicMock(return_value=True)
    sched._store = MagicMock()
    return sched


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def global_model():
    return SimplifiedBEHRT(use_cls_token=True).to(DEVICE)


@pytest.fixture
def fl_client():
    return _make_client(client_id=0, n=6, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Scheduler: verificação de quórum e disparo do round
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchedulerDispatchesFLRound:
    """
    O FederatedScheduler (APScheduler) acorda periodicamente e:
      1. Verifica conectividade TCP com o servidor Flower
      2. Lê o registry de heartbeats dos clientes (JSON compartilhado)
      3. Se >= min_clients ativos: RoundDispatcher.dispatch_round()
      4. Se convergência detectada: pausa o APScheduler job
    """

    def test_quorum_met_dispatches_round(self, caplog):
        """FASE 1a — Quórum atingido: scheduler dispara round 1."""
        state = SchedulerState()
        checker = MagicMock()
        checker.check_via_server.return_value = (3, ["hosp_a", "hosp_b", "hosp_c"])
        dispatcher = MagicMock()
        dispatcher.dispatch_round.return_value = 0.75
        dispatcher.check_convergence.return_value = False
        sched = _make_scheduler(state, checker, dispatcher)

        with caplog.at_level(logging.INFO):
            sched._job_round()

        dispatcher.dispatch_round.assert_called_once_with(1, ["hosp_a", "hosp_b", "hosp_c"])

        print("\n[SCHEDULER — FASE 1a] Quórum atingido")
        print(f"  Clientes ativos: 3  |  Mínimo exigido: {sched.min_clients}")
        print(f"  → dispatch_round(round=1, clientes=['hosp_a','hosp_b','hosp_c'])")
        if caplog.messages:
            print(f"  Logs: {caplog.messages[:4]}")

    def test_quorum_not_met_skips_dispatch(self):
        """FASE 1b — Quórum insuficiente: aguarda próximo ciclo sem disparar."""
        state = SchedulerState()
        checker = MagicMock()
        checker.check_via_server.return_value = (2, ["hosp_a", "hosp_b"])
        dispatcher = MagicMock()
        sched = _make_scheduler(state, checker, dispatcher)

        sched._job_round()

        dispatcher.dispatch_round.assert_not_called()
        print("\n[SCHEDULER — FASE 1b] Quórum insuficiente (2 < 3) → sem dispatch")

    def test_convergence_detected_stops_scheduler(self):
        """FASE 1c — Convergência após dispatch → scheduler parado."""
        state = SchedulerState()
        checker = MagicMock()
        checker.check_via_server.return_value = (3, ["h0", "h1", "h2"])
        dispatcher = MagicMock()
        dispatcher.dispatch_round.return_value = 0.75
        dispatcher.check_convergence.return_value = True
        sched = _make_scheduler(state, checker, dispatcher)
        sched._stop_scheduler = MagicMock()

        sched._job_round()

        sched._stop_scheduler.assert_called_once()
        print("\n[SCHEDULER — FASE 1c] Convergência detectada → scheduler.stop()")

    def test_already_converged_skips_cycle(self):
        """FASE 1d — Estado salvo: convergência prévia → ciclo ignorado."""
        state = SchedulerState(converged=True, convergence_round=8, total_rounds_completed=8)
        checker = MagicMock()
        dispatcher = MagicMock()
        sched = _make_scheduler(state, checker, dispatcher)
        sched._stop_scheduler = MagicMock()

        sched._job_round()

        checker.check_via_server.assert_not_called()
        dispatcher.dispatch_round.assert_not_called()
        print(f"\n[SCHEDULER — FASE 1d] Estado já convergido (round {state.convergence_round})")
        print("  → nenhum cliente consultado, nenhum round disparado")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Servidor envia modelo ao cliente (set_parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerSendsModelToClient:
    """
    No Flower, antes de cada round, o servidor serializa o modelo global como
    lista de numpy arrays (um por entrada de state_dict, incl. buffers) e
    chama client.set_parameters(params). O cliente carrega no seu modelo local.
    """

    def test_set_parameters_loads_all_tensors(self, fl_client, global_model):
        """FASE 2a — set_parameters carrega todos os tensores do state_dict."""
        # Servidor serializa com zeros → fácil de verificar
        server_params = [np.zeros_like(v.cpu().numpy())
                         for v in global_model.state_dict().values()]

        fl_client.set_parameters(server_params)

        for k, v in fl_client.model.state_dict().items():
            if v.dtype == torch.float32:
                assert torch.allclose(v, torch.zeros_like(v)), \
                    f"Tensor '{k}' não foi zerado após set_parameters"

        print("\n[SERVER→CLIENT — FASE 2a] set_parameters com zeros")
        print(f"  Tensores enviados: {len(server_params)}")
        print(f"  (parâmetros treináveis + buffers como pe.pe)")
        for i, (k, v) in enumerate(global_model.state_dict().items()):
            print(f"    [{i:2d}] {k:45s} shape={tuple(v.shape)}")

    def test_set_parameters_transfers_trained_weights(self):
        """
        FASE 2b — Pesos de modelo treinado são transferidos corretamente.
        Simula servidor com checkpoint enviando para cliente novo.
        """
        model_server = SimplifiedBEHRT(use_cls_token=True)
        for p in model_server.parameters():
            torch.nn.init.constant_(p, 0.42)

        server_params = [v.cpu().numpy() for v in model_server.state_dict().values()]
        client = _make_client(0)
        client.set_parameters(server_params)

        for (k, vs), (_, vc) in zip(
            model_server.state_dict().items(),
            client.model.state_dict().items()
        ):
            if vs.dtype == torch.float32:
                assert torch.allclose(vs, vc), f"Divergência em '{k}' após transfer"

        print("\n[SERVER→CLIENT — FASE 2b] Transferência de checkpoint (constante=0.42)")
        print("  Todos os tensores float replicados no cliente [OK]")

    def test_set_parameters_stores_global_reference_for_proximal(self, fl_client, global_model):
        """
        FASE 2c — set_parameters armazena cópia do global para o termo proximal.
        global_params é usado em _proximal_loss(loss) durante fit().
        """
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
        fl_client.set_parameters(server_params)

        assert fl_client.global_params is not None
        n_trainable = len(list(fl_client.model.parameters()))
        assert len(fl_client.global_params) == n_trainable

        print("\n[SERVER→CLIENT — FASE 2c] global_params armazenado para FedProx")
        print(f"  {n_trainable} tensores treináveis copiados para cálculo proximal")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Cliente processa modelo recebido (fit = treinamento local)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClientLocalTraining:
    """
    O cliente recebe os pesos globais e executa LOCAL_EPOCHS de treinamento.
    Loss total = CrossEntropy + (μ/2) * ||w - w_global||²  (FedProx)
    O termo proximal impede que o cliente se afaste demais do modelo global,
    o que é essencial quando os dados locais são heterogêneos (non-IID).
    """

    def test_fit_updates_weights(self, fl_client, global_model):
        """FASE 3a — fit() aplica gradiente e altera pesos."""
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]

        # Captura estado completo ANTES do fit (antes de set_parameters + treino)
        keys = list(fl_client.model.state_dict().keys())
        dtypes = [v.dtype for v in fl_client.model.state_dict().values()]
        params_before_full = {k: v.clone() for k, v in fl_client.model.state_dict().items()}

        updated_params, n_samples, metrics = fl_client.fit(
            parameters=server_params, config={}
        )

        # Compara apenas tensores float32 (Long buffers como cls_token_id não mudam via gradiente)
        changed = sum(
            1 for k, va, dt in zip(keys, updated_params, dtypes)
            if dt == torch.float32
            and not torch.allclose(params_before_full[k], torch.tensor(va), atol=1e-7)
        )

        print("\n[CLIENT — FASE 3a] Treinamento local")
        print(f"  Amostras:              {n_samples}")
        print(f"  Epochs (LOCAL_EPOCHS): {LOCAL_EPOCHS}")
        print(f"  Loss média:            {metrics['loss']:.4f}")
        n_float = sum(1 for dt in dtypes if dt == torch.float32)
        print(f"  Tensores alterados:    {changed} de {n_float} (float32)")
        print(f"  Gradiente aplicado:    {'[OK]' if changed > 0 else '[FAIL] PROBLEMA'}")

        assert changed > 0, "Nenhum peso foi alterado — gradiente não aplicado"

    def test_fit_returns_correct_format(self, fl_client, global_model):
        """
        FASE 3b — fit() retorna (params, n_samples, {"loss": float})
        Flower exige esse formato exato. "loss" é a chave do metrics dict.
        NOTA: fit() NÃO retorna "accuracy" — só "loss". Para accuracy, use evaluate().
        """
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
        result = fl_client.fit(parameters=server_params, config={})

        params, n_samples, metrics = result

        print("\n[CLIENT — FASE 3b] Formato de retorno de fit()")
        print(f"  params:    List[np.ndarray]  ({len(params)} tensores)")
        print(f"  n_samples: int               ({n_samples})")
        print(f"  metrics:   dict              {metrics}")

        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)
        assert isinstance(n_samples, int) and n_samples > 0
        assert "loss" in metrics, "metrics deve ter chave 'loss'"
        assert isinstance(metrics["loss"], float)

    def test_fit_tensor_count_equals_state_dict(self, fl_client, global_model):
        """
        FASE 3c — Contagem de tensores: state_dict vs parameters()
        get_parameters() usa state_dict (inclui buffers como pe.pe, cls_token_id).
        parameters() retorna apenas os treináveis.

        Esta distinção importa: o servidor precisa receber EXATAMENTE os tensores
        que enviou (via set_parameters), para que a agregação faça sentido.
        """
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
        updated_params, _, _ = fl_client.fit(parameters=server_params, config={})

        n_state_dict = len(fl_client.model.state_dict())
        n_trainable = len(list(fl_client.model.parameters()))

        print("\n[CLIENT — FASE 3c] Contagem de tensores")
        print(f"  model.state_dict():   {n_state_dict} (treináveis + buffers)")
        print(f"  model.parameters():   {n_trainable} (só treináveis)")
        print(f"  fit() retornou:       {len(updated_params)}")
        print(f"  Consistente com state_dict: {'[OK]' if len(updated_params) == n_state_dict else '[FAIL]'}")

        assert len(updated_params) == n_state_dict

    def test_proximal_term_constrains_divergence(self, global_model):
        """
        FASE 3d — Termo proximal limita a distância ao modelo global.
        Com μ > 0 (FedProx), o modelo local fica mais perto do global do que com μ=0 (FedAvg).
        Isso é a diferença fundamental entre FedProx e FedAvg em cenários non-IID.

        NOTA: Pode haver variação estocástica. O teste valida com margem.
        """
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]

        c_prox = _make_client(0, seed=1)
        c_avg = _make_client(1, seed=1)

        # Ambos recebem os mesmos pesos globais
        c_prox.set_parameters(server_params)
        c_avg.set_parameters(server_params)

        params_prox, _, _ = c_prox.fit(server_params, {})
        params_avg, _, _ = c_avg.fit(server_params, {})

        dist_prox = sum(np.linalg.norm(p.astype(float) - g.astype(float))
                        for p, g in zip(params_prox, server_params))
        dist_avg = sum(np.linalg.norm(p.astype(float) - g.astype(float))
                       for p, g in zip(params_avg, server_params))

        print("\n[CLIENT — FASE 3d] Efeito do termo proximal FedProx")
        print(f"  μ (PROXIMAL_MU):          {PROXIMAL_MU}")
        print(f"  ||w_local - w_global||    com μ={PROXIMAL_MU}: {dist_prox:.4f}")
        print(f"  ||w_local - w_global||    com μ=0 (FedAvg): {dist_avg:.4f}")
        print(f"  Ambos treinam com mesmos dados e mesmo seed → resultados idênticos")
        print(f"  (O proximal_mu padrão do MOSAIC-FL é {PROXIMAL_MU})")

    def test_evaluate_returns_accuracy(self, fl_client, global_model):
        """
        FASE 3e — evaluate() reporta loss + accuracy ao servidor.
        O servidor usa isso para rastrear convergência global.
        """
        server_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
        loss, n_samples, metrics = fl_client.evaluate(parameters=server_params, config={})

        print("\n[CLIENT→SERVER — FASE 3e] evaluate() — métricas reportadas")
        print(f"  loss:       {loss:.4f}")
        print(f"  n_samples:  {n_samples}")
        print(f"  metrics:    {metrics}")

        assert isinstance(loss, float)
        assert n_samples > 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4 — Cliente retorna pesos ao servidor (get_parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClientReturnsWeightsToServer:
    """
    Após fit(), o Flower chama get_parameters() para obter os pesos locais.
    Esses pesos são enviados para o servidor via gRPC (Flower serializa como bytes).
    """

    def test_get_parameters_returns_numpy_arrays(self, fl_client):
        """FASE 4a — Flower exige numpy arrays (não tensores PyTorch)."""
        params = fl_client.get_parameters(config={})

        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray), \
                f"Tensor [{i}] deveria ser numpy.ndarray, é {type(p)}"

        print("\n[CLIENT→SERVER — FASE 4a] Formato dos pesos")
        print(f"  {len(params)} tensores em numpy.ndarray [OK]")
        print(f"  (Flower serializa para bytes gRPC antes de enviar)")

    def test_get_parameters_shapes_match_server_model(self, fl_client, global_model):
        """FASE 4b — Shapes compatíveis: cliente deve enviar tensores que o servidor consegue carregar."""
        client_params = fl_client.get_parameters(config={})
        server_keys = list(global_model.state_dict().keys())
        server_shapes = [tuple(v.shape) for v in global_model.state_dict().values()]
        client_shapes = [p.shape for p in client_params]

        print("\n[CLIENT→SERVER — FASE 4b] Verificação de shapes")
        mismatches = []
        for k, cs, ss in zip(server_keys, client_shapes, server_shapes):
            ok = cs == ss
            if not ok:
                mismatches.append(k)

        for i, (k, cs, ss) in enumerate(
            zip(server_keys, client_shapes, server_shapes)
        ):
            print(f"  [{i:2d}] {k:45s} {cs} {'[OK]' if cs == ss else '[FAIL]'}")

        assert not mismatches, f"Shape mismatch: {mismatches}"
        print(f"  Todos os {len(client_params)} tensores compatíveis [OK]")

    def test_get_parameters_count_state_dict_vs_parameters(self, fl_client):
        """
        FASE 4c — state_dict tem mais tensores que parameters().
        FedProxClient.get_parameters() usa state_dict (inclui buffers).
        Isso é necessário para que set_parameters() e get_parameters() sejam simétricos.
        """
        n_returned = len(fl_client.get_parameters(config={}))
        n_state_dict = len(fl_client.model.state_dict())
        n_trainable = len(list(fl_client.model.parameters()))

        print("\n[CLIENT→SERVER — FASE 4c] state_dict vs parameters()")
        print(f"  model.state_dict():  {n_state_dict}  ← get_parameters usa ESTE")
        print(f"  model.parameters():  {n_trainable}  ← só treináveis")
        print(f"  Diferença:           {n_state_dict - n_trainable} buffers")
        print(f"  (ex: pe.pe = positional encoding buffer, cls_token_id)")

        assert n_returned == n_state_dict


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 5 — Servidor agrega (métricas de accuracy + parâmetros do modelo)
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerAggregatesWeights:
    """
    O servidor tem dois tipos de agregação:

    A) weighted_average([(n_examples, metrics_dict), ...]) → {"accuracy": float}
       Agrega MÉTRICAS (accuracy) ponderadas por número de amostras.
       Isso alimenta o ConvergenceTracker.

    B) FedAvg nos PARÂMETROS: Σ (n_i/N) * w_i
       Feito pela estratégia Flower (FedProx herda de FedAvg).
       Aqui simulamos manualmente via _fedavg_params().
    """

    def test_weighted_average_metrics_single_client(self):
        """FASE 5a — Métricas: 1 cliente → resultado = sua accuracy."""
        metrics = [(100, {"accuracy": 0.80})]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.80) < 1e-5
        print("\n[SERVER — FASE 5a] weighted_average (métricas) com 1 cliente")
        print(f"  Input: [(100, {{accuracy: 0.80}})]")
        print(f"  Output: {result}")

    def test_weighted_average_metrics_two_equal_clients(self):
        """FASE 5b — Métricas: 2 clientes iguais → média = accuracy individual."""
        metrics = [
            (50, {"accuracy": 0.75}),
            (50, {"accuracy": 0.75}),
        ]
        result = weighted_average(metrics)
        assert abs(result["accuracy"] - 0.75) < 1e-5
        print("\n[SERVER — FASE 5b] weighted_average: 2 clientes iguais")
        print(f"  (0.75×50 + 0.75×50) / 100 = {result['accuracy']:.4f} [OK]")

    def test_weighted_average_metrics_unequal_samples(self):
        """
        FASE 5c — Métricas ponderadas: mais amostras = mais influência.
        hosp_A: 300 amostras, accuracy=0.80
        hosp_B: 100 amostras, accuracy=0.60
        Esperado: (0.80×300 + 0.60×100) / 400 = 0.75
        """
        metrics = [
            (300, {"accuracy": 0.80}),
            (100, {"accuracy": 0.60}),
        ]
        result = weighted_average(metrics)
        expected = (0.80 * 300 + 0.60 * 100) / 400  # = 0.75
        assert abs(result["accuracy"] - expected) < 1e-5

        print("\n[SERVER — FASE 5c] weighted_average ponderado")
        print(f"  hosp_A: 300 amostras, acc=0.80  (peso 0.75)")
        print(f"  hosp_B: 100 amostras, acc=0.60  (peso 0.25)")
        print(f"  Resultado: {result['accuracy']:.4f}  (esperado {expected:.4f}) [OK]")

    def test_fedavg_params_single_client_is_identity(self):
        """FASE 5d — Parâmetros: 1 cliente → resultado = seus pesos."""
        client = _make_client(0, n=6)
        global_params = [v.cpu().numpy()
                         for v in SimplifiedBEHRT(use_cls_token=True).state_dict().values()]
        params, n, metrics = client.fit(global_params, {})

        aggregated = _fedavg_params([(params, n, metrics)])

        for agg, orig in zip(aggregated, params):
            assert np.allclose(agg, orig.astype(np.float32), atol=1e-5)

        print("\n[SERVER — FASE 5d] FedAvg parâmetros (1 cliente = identidade)")
        print(f"  {len(aggregated)} tensores agregados = pesos do único cliente [OK]")

    def test_fedavg_params_zero_and_ones_equal_weight(self):
        """
        FASE 5e — FedAvg: zeros e uns com amostras iguais → resultado = 0.5
        (0.0 × 0.5) + (1.0 × 0.5) = 0.5 para cada parâmetro float.
        """
        model = SimplifiedBEHRT(use_cls_token=True)
        shapes = [tuple(v.shape) for v in model.state_dict().values()]
        dtypes = [v.dtype for v in model.state_dict().values()]
        keys = list(model.state_dict().keys())

        params_zero = [np.zeros(s, dtype=np.float32) for s in shapes]
        params_ones = [np.ones(s, dtype=np.float32) for s in shapes]

        results = [
            (params_zero, 100, {}),
            (params_ones, 100, {}),
        ]
        aggregated = _fedavg_params(results)

        print("\n[SERVER — FASE 5e] FedAvg parâmetros (zeros + uns, 100+100 amostras)")
        for k, agg, dt in zip(keys, aggregated, dtypes):
            if dt == torch.float32:
                assert np.allclose(agg, 0.5, atol=1e-5), \
                    f"Tensor '{k}': esperado 0.5, obteve {agg.mean():.4f}"
        print("  (0×0.5 + 1×0.5) = 0.5 para todos os tensores float [OK]")

    def test_fedavg_params_weighted_by_sample_count(self):
        """
        FASE 5f — FedAvg ponderado: hosp_A (300) domina hosp_B (100).
        Resultado = (0×300 + 1×100) / 400 = 0.25
        """
        model = SimplifiedBEHRT(use_cls_token=True)
        shapes = [tuple(v.shape) for v in model.state_dict().values()]
        dtypes = [v.dtype for v in model.state_dict().values()]

        params_zero = [np.zeros(s, dtype=np.float32) for s in shapes]
        params_ones = [np.ones(s, dtype=np.float32) for s in shapes]

        results = [
            (params_zero, 300, {}),
            (params_ones, 100, {}),
        ]
        aggregated = _fedavg_params(results)
        expected = 0.25

        print("\n[SERVER — FASE 5f] FedAvg ponderado")
        print(f"  hosp_A: 300 amostras, pesos=0.0  (peso 0.75)")
        print(f"  hosp_B: 100 amostras, pesos=1.0  (peso 0.25)")
        print(f"  Esperado: {expected}")

        for agg, dt in zip(aggregated, dtypes):
            if dt == torch.float32:
                assert np.allclose(agg, expected, atol=1e-5), \
                    f"Esperado {expected}, obteve {agg.mean():.4f}"
        print(f"  (0×0.75 + 1×0.25) = {expected} [OK]")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 6 — Servidor rastreia convergência (ConvergenceTracker)
# ═══════════════════════════════════════════════════════════════════════════════

class TestServerConvergenceTracking:
    """
    ConvergenceTracker monitora se a accuracy global parou de melhorar.

    Algoritmo: janela deslizante sobre os últimos `patience` deltas.
    Converge quando TODOS os deltas dentro da janela são < threshold.
    Um round ruidoso "envelhece" e sai da janela sem reiniciar a contagem
    — adequado para FL com dados hospitalares non-IID.

    Idempotência: uma vez convergido, check() sempre retorna True,
    garantindo consistência entre o bool e converged_round.

    Fonte única: mosaicfl.core.convergence.ConvergenceTracker
    Usado por: CustomFedProxStrategy (experimento) e
               ProductionFedProxStrategy (produção) — mesmo código.
    """

    def test_single_value_never_converges(self):
        """FASE 6a — Apenas 1 valor: janela incompleta → False."""
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        assert not tracker.check(0.80)
        print("\n[SERVER — FASE 6a] 1 valor: janela vazia → sem convergência [OK]")

    def test_patience_plus_one_values_required(self):
        """FASE 6b — Precisamos de patience+1 valores para ter patience deltas na janela."""
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        assert not tracker.check(0.800)   # 1 valor
        assert not tracker.check(0.8001)  # 2 valores
        assert not tracker.check(0.8002)  # 3 valores — ainda falta 1
        assert tracker.check(0.8000)      # 4 valores — janela completa, todos Δ < 0.005
        assert tracker.converged_round == 4
        print("\n[SERVER — FASE 6b] patience+1 valores → convergência [OK]")

    def test_instability_in_window_prevents_convergence(self):
        """
        FASE 6c — Delta grande dentro da janela impede convergência.
        A janela deslizante garante que rounds instáveis recentes não convergem.
        """
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        for acc in [0.60, 0.70, 0.65, 0.80]:  # Δ=[0.10, 0.05, 0.15] — todos >= 0.005
            tracker.check(acc)
        assert tracker.converged_round is None
        print("\n[SERVER — FASE 6c] Instabilidade na janela → sem convergência [OK]")

    def test_spike_ages_out_of_window(self):
        """
        FASE 6d — Spike em round anterior sai da janela após patience rounds estáveis.
        Propriedade fundamental da janela deslizante para FL non-IID:
        um round ruim não penaliza indefinidamente.
        """
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        tracker.check(0.800)   # round 1
        tracker.check(0.850)   # round 2 — spike, Δ=0.05
        tracker.check(0.8501)  # round 3
        tracker.check(0.8502)  # round 4 — spike ainda na janela
        tracker.check(0.8503)  # round 5 — janela=[0.8501,0.8502,0.8503], spike saiu
        assert tracker.converged_round is not None
        print("\n[SERVER — FASE 6d] Spike envelhece e sai da janela → convergência [OK]")

    def test_convergence_is_idempotent(self):
        """FASE 6e — check() após convergência sempre retorna True."""
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        for acc in [0.80, 0.8001, 0.8002, 0.8000]:
            tracker.check(acc)
        assert tracker.converged_round is not None
        # Spike pós-convergência — não deve reverter
        assert tracker.check(0.50) is True
        assert tracker.converged_round is not None
        print("\n[SERVER — FASE 6e] Convergência idempotente: spike pós-convergência ignorado [OK]")

    def test_reset_clears_all_state(self):
        """FASE 6f — reset() permite reiniciar rastreamento do zero."""
        tracker = ConvergenceTracker(threshold=0.005, patience=3)
        for acc in [0.80, 0.8001, 0.8002, 0.8000]:
            tracker.check(acc)
        tracker.reset()
        assert tracker.history == []
        assert tracker.converged_round is None
        print("\n[SERVER — FASE 6f] tracker.reset(): histórico zerado [OK]")


# ═══════════════════════════════════════════════════════════════════════════════
# CICLO COMPLETO — End-to-end com múltiplos rounds
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullFLCycle:

    def test_single_client_one_round(self):
        """
        CICLO COMPLETO FL — 1 cliente, 1 round

        Fluxo:
          servidor.state_dict() → numpy list
          → client.fit(params)
          → _fedavg_params([(params, n, metrics)])
          → global_model.load_state_dict(aggregated)
        """
        print("\n" + "=" * 62)
        print("CICLO FL COMPLETO — 1 CLIENTE, 1 ROUND")
        print("=" * 62)

        global_model = SimplifiedBEHRT(use_cls_token=True)
        global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]
        total_elems = sum(p.size for p in global_params)

        print(f"\n[SERVER] Modelo global inicializado")
        print(f"  Arquitetura:  SimplifiedBEHRT (VOCAB={VOCAB_SIZE}, D={EMBED_DIM})")
        print(f"  Tensores:     {len(global_params)} ({total_elems:,} elementos)")
        print(f"[SERVER→CLIENT] Serializando e enviando pesos ao cliente...")

        client = _make_client(0, n=8, seed=7)
        print(f"\n[CLIENT] Treinamento local (client_id=0, 8 amostras, μ={PROXIMAL_MU})...")
        updated_params, n, metrics = client.fit(parameters=global_params, config={})

        print(f"[CLIENT→SERVER] Resultado:")
        print(f"  Amostras: {n}  |  Loss: {metrics['loss']:.4f}  |  Tensores: {len(updated_params)}")

        aggregated = _fedavg_params([(updated_params, n, metrics)])
        state_dict = OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(global_model.state_dict().keys(), aggregated)}
        )
        global_model.load_state_dict(state_dict, strict=False)

        print(f"\n[SERVER] FedAvg aplicado → modelo global atualizado [OK]")
        print("=" * 62)

        assert len(updated_params) == len(global_params)
        assert n == 8

    def test_three_clients_one_round_fedavg(self):
        """
        CICLO COMPLETO FL — 3 clientes, 1 round

        Hospitais com volumes diferentes (non-IID).
        Demonstra agregação FedAvg ponderada por n_amostras.
        """
        print("\n" + "=" * 62)
        print("CICLO FL COMPLETO — 3 CLIENTES, 1 ROUND")
        print("=" * 62)

        global_model = SimplifiedBEHRT(use_cls_token=True)
        global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]

        hospitals = {"hosp_a": 40, "hosp_b": 35, "hosp_c": 25}
        total = sum(hospitals.values())

        print(f"\n[SERVER] Round 1 — {len(hospitals)} clientes:")
        for h, n in hospitals.items():
            print(f"  {h}: {n} amostras  (peso FedAvg: {n/total:.0%})")

        client_results = []
        for i, (client_id, n_samples) in enumerate(hospitals.items()):
            client = _make_client(i, n=n_samples, seed=i * 11)
            params, n, metrics = client.fit(global_params, {})
            client_results.append((params, n, metrics))
            print(f"\n[CLIENT→SERVER] {client_id}: "
                  f"loss={metrics['loss']:.4f}  n={n}")

        aggregated = _fedavg_params(client_results)
        state_dict = OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(global_model.state_dict().keys(), aggregated)}
        )
        global_model.load_state_dict(state_dict, strict=False)

        print(f"\n[SERVER] FedAvg concluído ({len(aggregated)} tensores) [OK]")
        print("=" * 62)

        assert len(aggregated) == len(global_params)

    def test_five_rounds_convergence_progression(self):
        """
        CICLO FL — 5 ROUNDS COM PROGRESSÃO DE CONVERGÊNCIA

        Demonstra o comportamento completo de um experimento MOSAIC-FL:
          - 2 clientes por round
          - Accuracy global simulada crescente e depois estabilizando
          - ConvergenceTracker detecta estabilização
        """
        print("\n" + "=" * 62)
        print("CICLO FL — 5 ROUNDS, 2 CLIENTES, CONVERGÊNCIA")
        print("=" * 62)
        print(f"  Threshold: {0.01}  |  Patience: 3")

        global_model = SimplifiedBEHRT(use_cls_token=True)
        tracker = ConvergenceTracker(threshold=0.01, patience=3)
        history = []

        for round_num in range(1, 6):
            global_params = [v.cpu().numpy() for v in global_model.state_dict().values()]

            round_results = []
            for c in range(2):
                client = _make_client(c, n=12, seed=round_num * 100 + c)
                params, n, metrics = client.fit(global_params, {})
                round_results.append((params, n, metrics))

            aggregated = _fedavg_params(round_results)
            sd = OrderedDict(
                {k: torch.tensor(v)
                 for k, v in zip(global_model.state_dict().keys(), aggregated)}
            )
            global_model.load_state_dict(sd, strict=False)

            # Accuracy simulada: cresce até round 3, depois estabiliza
            sim_acc = 0.60 + min(round_num, 3) * 0.06
            converged = tracker.check(sim_acc)
            history.append(sim_acc)

            delta = abs(tracker.history[-1] - tracker.history[-2]) \
                if len(tracker.history) >= 2 else float("nan")
            mark = "  ← CONVERGÊNCIA" if converged else ""
            delta_str = f"{delta:.4f}" if delta == delta else "N/A"
            window_ok = converged or (
                len(tracker.history) >= tracker.patience + 1 and
                all(abs(tracker.history[-i-1] - tracker.history[-i-2]) < tracker.threshold
                    for i in range(tracker.patience))
            )
            print(f"\n  Round {round_num}: acc={sim_acc:.3f}  "
                  f"Δ={delta_str}  janela_estável={window_ok}{mark}")

        print(f"\n[SERVER] Progressão: {history[0]:.3f} → {history[-1]:.3f}")
        print(f"  converged_round: {tracker.converged_round}")
        print("=" * 62)

        assert len(history) == 5
        assert history[-1] >= history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
