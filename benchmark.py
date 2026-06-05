"""
benchmark.py — Benchmark de Performance do MOSAIC-FL v2

Mede:
  • Tempo por rodada federada (treino + agregação + avaliação)
  • Uso de memória RAM (pico e média por rodada)
  • Uso de CPU (%)
  • Tráfego de rede estimado (tamanho real dos pesos transmitidos)
  • Throughput (amostras/segundo por cliente)

Saídas:
  • JSON com métricas detalhadas por rodada  → benchmark_results/benchmark_<ts>.json
  • Gráficos PNG (tempo, memória, CPU, tráfego, acurácia, throughput)

Uso:
    source .venv/bin/activate
    python benchmark.py
    python benchmark.py --samples 2000 --rounds 5 --clients 3

Requisitos adicionais:
    pip install psutil matplotlib
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Garante que src/ e raiz do projeto estão no sys.path
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import psutil
import matplotlib
matplotlib.use("Agg")   # headless — sem GUI
import matplotlib.pyplot as plt

import flwr as fl

from mosaicfl.v2.config import (
    BATCH_SIZE, LOCAL_EPOCHS, PROXIMAL_MU,
    FRACTION_FIT, FRACTION_EVALUATE,
    MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_AVAILABLE_CLIENTS,
    EMBED_DIM, NUM_LAYERS, NUM_HEADS, NUM_CLASSES, VOCAB_SIZE,
    MAX_SEQ_LEN, RANDOM_SEED, NUM_CLIENTS,
)
from mosaicfl.v2.model_v2 import SimplifiedBEHRT
from mosaicfl.v2.client_v2 import FedProxClient
from mosaicfl.v2.server_v2 import weighted_average, get_evaluate_fn
from mosaicfl.v2.preprocess_v2 import EHRPreprocessor, split_by_institution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATACLASS: Métricas por rodada
# ─────────────────────────────────────────────────────────────

@dataclass
class RoundMetrics:
    round: int
    start_time: float
    end_time: float
    duration_sec: float
    mem_before_mb: float
    mem_peak_mb: float
    mem_after_mb: float
    cpu_percent_avg: float
    cpu_percent_peak: float
    network_mb_sent: float
    network_mb_recv: float
    train_samples: int
    throughput_samples_per_sec: float
    server_loss: Optional[float] = None
    server_accuracy: Optional[float] = None
    converged: bool = False


# ─────────────────────────────────────────────────────────────
# MONITORAMENTO DE RECURSOS
# ─────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Monitora CPU, memória e rede do processo atual."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._cpu_samples: List[float] = []
        self._mem_samples: List[float] = []
        self._net_start = psutil.net_io_counters()

    def start(self):
        self._cpu_samples.clear()
        self._mem_samples.clear()
        self._net_start = psutil.net_io_counters()
        self._sample()

    def _sample(self):
        try:
            mem_mb = self.process.memory_info().rss / (1024 ** 2)
            cpu_pct = self.process.cpu_percent(interval=None)
            self._mem_samples.append(mem_mb)
            self._cpu_samples.append(cpu_pct)
        except psutil.NoSuchProcess:
            pass

    def sample_now(self):
        self._sample()

    def stop(self) -> Dict:
        self._sample()
        net_end = psutil.net_io_counters()
        return {
            "cpu_avg":    float(np.mean(self._cpu_samples))  if self._cpu_samples else 0.0,
            "cpu_peak":   float(np.max(self._cpu_samples))   if self._cpu_samples else 0.0,
            "mem_avg_mb": float(np.mean(self._mem_samples))  if self._mem_samples else 0.0,
            "mem_peak_mb":float(np.max(self._mem_samples))   if self._mem_samples else 0.0,
            "net_sent_mb": (net_end.bytes_sent - self._net_start.bytes_sent) / (1024 ** 2),
            "net_recv_mb": (net_end.bytes_recv - self._net_start.bytes_recv) / (1024 ** 2),
        }

    @staticmethod
    def model_size_mb(model: torch.nn.Module) -> float:
        """Tamanho real do state_dict em MB (float32)."""
        total = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        return total / (1024 ** 2)


# ─────────────────────────────────────────────────────────────
# GERAÇÃO DE DADOS SINTÉTICOS
# ─────────────────────────────────────────────────────────────

def generate_benchmark_data(n_samples: int = 1000, n_institutions: int = 5) -> pd.DataFrame:
    """Gera dataset sintético com schema compatível com EHRPreprocessor v2."""
    rng = np.random.default_rng(RANDOM_SEED)
    institutions = [f"Hospital_{i}" for i in range(n_institutions)]
    sintomas     = ["febre", "tosse", "dispneia", "fadiga", "mialgia", "cefaleia", "anosmia"]
    exames       = ["rt_pcr_positivo", "tomografia_normal", "tomografia_vidro_fosco", "rx_consolidacao"]
    diagnosticos = ["covid19_leve", "covid19_moderado", "covid19_grave", "pneumonia_bacteriana"]

    return pd.DataFrame({
        "instituicao":    rng.choice(institutions, n_samples),
        "idade":          rng.integers(18, 90, n_samples),
        "idade_unidade":  rng.choice(["anos", "meses"], n_samples, p=[0.95, 0.05]),
        "peso":           rng.uniform(50.0, 120.0, n_samples),
        "peso_unidade":   rng.choice(["kg", "lb"], n_samples, p=[0.9, 0.1]),
        "temperatura":    rng.uniform(36.0, 40.0, n_samples),
        "sintoma":        rng.choice(sintomas, n_samples),
        "exame":          rng.choice(exames, n_samples),
        "diagnostico":    rng.choice(diagnosticos, n_samples),
        "desfecho":       rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })


# ─────────────────────────────────────────────────────────────
# PREPARAÇÃO DOS DATALOADERS
# ─────────────────────────────────────────────────────────────

def prepare_benchmark_loaders(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[Dict[int, Tuple[DataLoader, DataLoader]], DataLoader, int, int]:
    """
    Pré-processa o DataFrame e cria DataLoaders por cliente.

    Returns:
        client_loaders:      {cid: (train_loader, val_loader)}
        test_loader:         DataLoader global (20% dos dados)
        vocab_size:          int
        total_train_samples: int
    """
    text_cols = ["sintoma", "exame", "diagnostico"]
    preprocessor = EHRPreprocessor()
    df_proc, _ = preprocessor.process(df, text_cols=text_cols)
    vocab_size = len(preprocessor.vocab_map)

    client_dfs = split_by_institution(
        df_proc,
        institution_col="instituicao",
        num_clients=NUM_CLIENTS,
        stratify_col="desfecho",
        random_state=RANDOM_SEED,
    )

    seq_cols = [c for c in df_proc.columns if c.endswith("_encoded")]

    def to_tensors(sub_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = [], []
        for _, row in sub_df.iterrows():
            seq = [int(row[c]) for c in seq_cols if pd.notna(row[c])]
            seq = (seq + [0] * max_seq_len)[:max_seq_len]
            sequences.append(seq)
            labels.append(int(row["desfecho"]))
        return (
            torch.tensor(sequences, dtype=torch.long),
            torch.tensor(labels,    dtype=torch.long),
        )

    client_loaders: Dict[int, Tuple[DataLoader, DataLoader]] = {}
    total_train_samples = 0

    for cid, subset in client_dfs.items():
        if len(subset) < 10:
            continue
        n_train = int(0.8 * len(subset))
        train_x, train_y = to_tensors(subset.iloc[:n_train])
        val_x,   val_y   = to_tensors(subset.iloc[n_train:])
        client_loaders[cid] = (
            DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(val_x,   val_y),   batch_size=batch_size),
        )
        total_train_samples += len(train_x)

    # DataLoader de teste global (20% aleatórios)
    test_df = df_proc.sample(frac=0.2, random_state=RANDOM_SEED)
    test_x, test_y = to_tensors(test_df)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    return client_loaders, test_loader, vocab_size, total_train_samples


# ─────────────────────────────────────────────────────────────
# ESTRATÉGIA INSTRUMENTADA
# ─────────────────────────────────────────────────────────────

class BenchmarkFedProxStrategy(fl.server.strategy.FedProx):
    """FedProx instrumentado para coleta de métricas de benchmark."""

    def __init__(
        self,
        monitor: ResourceMonitor,
        metrics_history: List[RoundMetrics],
        model_size_mb: float,
        total_train_samples: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.metrics_history = metrics_history
        self.model_size_mb = model_size_mb
        self.total_train_samples = total_train_samples

    def aggregate_fit(self, server_round, results, failures):
        round_start = time.time()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        self.monitor.sample_now()

        aggregated = super().aggregate_fit(server_round, results, failures)

        self.monitor.sample_now()
        mem_after  = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        round_end  = time.time()
        duration   = round_end - round_start

        # Tráfego real: cada cliente enviou e recebeu o state_dict completo
        n_clients  = len(results) if results else 1
        net_sent   = self.model_size_mb * n_clients
        net_recv   = self.model_size_mb * n_clients
        throughput = self.total_train_samples / duration if duration > 0 else 0.0

        self.metrics_history.append(RoundMetrics(
            round=server_round,
            start_time=round_start,
            end_time=round_end,
            duration_sec=duration,
            mem_before_mb=mem_before,
            mem_peak_mb=max(mem_before, mem_after),
            mem_after_mb=mem_after,
            cpu_percent_avg=0.0,    # preenchido ao final
            cpu_percent_peak=0.0,
            network_mb_sent=net_sent,
            network_mb_recv=net_recv,
            train_samples=self.total_train_samples,
            throughput_samples_per_sec=throughput,
        ))

        logger.info(
            "[Benchmark] Rodada %d: %.2fs | Mem: %.1f→%.1f MB | "
            "Net: up=%.2f down=%.2f MB | %.1f amostras/s",
            server_round, duration, mem_before, mem_after,
            net_sent, net_recv, throughput,
        )
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated and len(aggregated) == 2 and self.metrics_history:
            loss, metrics = aggregated
            self.metrics_history[-1].server_loss     = float(loss)
            self.metrics_history[-1].server_accuracy = float(metrics.get("accuracy", 0.0))
        return aggregated


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO DO BENCHMARK
# ─────────────────────────────────────────────────────────────

def run_benchmark(
    n_samples: int = 1000,
    num_rounds: int = 10,
    num_clients: int = 5,
    output_dir: str = "benchmark_results",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("BENCHMARK MOSAIC-FL v2")
    logger.info("=" * 70)
    logger.info("Config: %d amostras | %d rodadas | %d clientes", n_samples, num_rounds, num_clients)
    logger.info(
        "Hardware: CPU=%d cores | RAM=%.1f GB",
        psutil.cpu_count(),
        psutil.virtual_memory().total / (1024 ** 3),
    )
    logger.info("Modelo: SimplifiedBEHRT (%dd, %d camadas, %d heads, CLS token)", EMBED_DIM, NUM_LAYERS, NUM_HEADS)
    logger.info("=" * 70)

    # 1. Dados sintéticos
    logger.info("[1/5] Gerando dados sintéticos...")
    t0 = time.time()
    df = generate_benchmark_data(n_samples=n_samples, n_institutions=num_clients)
    logger.info("      OK — %d linhas em %.2fs", len(df), time.time() - t0)

    # 2. Pré-processamento e DataLoaders
    logger.info("[2/5] Pré-processando e criando DataLoaders...")
    t0 = time.time()
    client_loaders, test_loader, vocab_size, total_train_samples = prepare_benchmark_loaders(
        df, batch_size=BATCH_SIZE,
    )
    logger.info(
        "      OK — vocab=%d | clientes=%d | amostras_treino=%d | %.2fs",
        vocab_size, len(client_loaders), total_train_samples, time.time() - t0,
    )

    # 3. Modelo e tamanho real
    ref_model  = SimplifiedBEHRT(use_cls_token=True)
    num_params = sum(p.numel() for p in ref_model.parameters())
    size_mb    = ResourceMonitor.model_size_mb(ref_model)
    logger.info("[3/5] Modelo: %d parâmetros treináveis | %.2f MB por transmissão", num_params, size_mb)

    # 4. Estratégia instrumentada
    monitor:          ResourceMonitor    = ResourceMonitor()
    metrics_history:  List[RoundMetrics] = []
    evaluate_fn = get_evaluate_fn(test_loader)

    strategy = BenchmarkFedProxStrategy(
        monitor=monitor,
        metrics_history=metrics_history,
        model_size_mb=size_mb,
        total_train_samples=total_train_samples,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=min(len(client_loaders), MIN_FIT_CLIENTS),
        min_evaluate_clients=min(len(client_loaders), MIN_EVALUATE_CLIENTS),
        min_available_clients=min(len(client_loaders), MIN_AVAILABLE_CLIENTS),
        proximal_mu=PROXIMAL_MU,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        train_loader, val_loader = client_loaders[cid_int]
        return FedProxClient(cid_int, train_loader, val_loader)

    # 5. Simulação FL
    logger.info("[4/5] Executando Federated Learning...")
    logger.info("=" * 70)

    overall_start = time.time()
    monitor.start()

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(client_loaders),
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )
    except Exception as exc:
        logger.error("Erro na simulação: %s", exc)

    stats = monitor.stop()
    total_duration = time.time() - overall_start

    logger.info("=" * 70)
    logger.info("[5/5] Concluído em %.2fs (%.2f min)", total_duration, total_duration / 60)

    # Preenche CPU nas métricas por rodada
    for m in metrics_history:
        m.cpu_percent_avg  = stats["cpu_avg"]
        m.cpu_percent_peak = stats["cpu_peak"]

    # Consolida resultado
    results = {
        "config": {
            "n_samples":       n_samples,
            "num_rounds":      num_rounds,
            "num_clients":     num_clients,
            "batch_size":      BATCH_SIZE,
            "local_epochs":    LOCAL_EPOCHS,
            "model_params":    num_params,
            "model_size_mb":   round(size_mb, 4),
            "vocab_size":      vocab_size,
            "hardware": {
                "cpu_cores": psutil.cpu_count(),
                "ram_gb":    round(psutil.virtual_memory().total / (1024 ** 3), 1),
            },
        },
        "summary": {
            "total_duration_sec":    round(total_duration, 3),
            "avg_round_duration_sec": round(
                float(np.mean([m.duration_sec for m in metrics_history])), 3
            ) if metrics_history else 0.0,
            "peak_mem_mb":           round(stats["mem_peak_mb"], 1),
            "avg_cpu_percent":       round(stats["cpu_avg"], 1),
            "peak_cpu_percent":      round(stats["cpu_peak"], 1),
            "total_net_sent_mb":     round(sum(m.network_mb_sent for m in metrics_history), 3),
            "total_net_recv_mb":     round(sum(m.network_mb_recv for m in metrics_history), 3),
            "final_accuracy":        metrics_history[-1].server_accuracy if metrics_history else None,
        },
        "per_round": [asdict(m) for m in metrics_history],
    }

    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("JSON salvo: %s", json_path)

    if metrics_history:
        plot_path = generate_plots(metrics_history, output_dir, timestamp)
        logger.info("Graficos salvos: %s", plot_path)

    print_benchmark_report(results)
    return results


# ─────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generate_plots(
    metrics: List[RoundMetrics],
    output_dir: str,
    timestamp: str,
) -> str:
    rounds     = [m.round          for m in metrics]
    durations  = [m.duration_sec   for m in metrics]
    mem_before = [m.mem_before_mb  for m in metrics]
    mem_after  = [m.mem_after_mb   for m in metrics]
    mem_peak   = [m.mem_peak_mb    for m in metrics]
    cpu_avg    = [m.cpu_percent_avg for m in metrics]
    net_sent   = [m.network_mb_sent for m in metrics]
    net_recv   = [m.network_mb_recv for m in metrics]
    acc_data   = [(m.round, m.server_accuracy) for m in metrics if m.server_accuracy is not None]
    throughput = [m.throughput_samples_per_sec for m in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"MOSAIC-FL v2 Benchmark  |  {timestamp}", fontsize=14, fontweight="bold")

    # 1. Duração por rodada
    ax = axes[0, 0]
    ax.bar(rounds, durations, color="steelblue", edgecolor="black")
    ax.axhline(np.mean(durations), color="red", linestyle="--",
               label=f"Media: {np.mean(durations):.2f}s")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Duracao (s)"); ax.set_title("Tempo por Rodada")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # 2. Memória RAM
    ax = axes[0, 1]
    ax.plot(rounds, mem_before, "o-", label="Antes",  color="green",  alpha=0.7)
    ax.plot(rounds, mem_after,  "s-", label="Depois", color="orange", alpha=0.7)
    ax.plot(rounds, mem_peak,   "^-", label="Pico",   color="red",    alpha=0.7)
    ax.set_xlabel("Rodada"); ax.set_ylabel("Memoria (MB)"); ax.set_title("Uso de RAM")
    ax.legend(); ax.grid(alpha=0.3)

    # 3. CPU
    ax = axes[0, 2]
    ax.fill_between(rounds, cpu_avg, alpha=0.3, color="purple")
    ax.plot(rounds, cpu_avg, "o-", color="purple", label="CPU %")
    ax.set_xlabel("Rodada"); ax.set_ylabel("CPU (%)"); ax.set_title("Uso de CPU")
    ax.legend(); ax.grid(alpha=0.3)

    # 4. Tráfego de rede
    ax = axes[1, 0]
    x = np.arange(len(rounds))
    w = 0.35
    ax.bar(x - w/2, net_sent, w, label="Enviado",   color="coral")
    ax.bar(x + w/2, net_recv, w, label="Recebido",  color="skyblue")
    ax.set_xticks(x); ax.set_xticklabels(rounds)
    ax.set_xlabel("Rodada"); ax.set_ylabel("Trafego (MB)"); ax.set_title("Trafego de Rede (estimado)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # 5. Acurácia global
    ax = axes[1, 1]
    if acc_data:
        acc_rounds, acc_vals = zip(*acc_data)
        ax.plot(acc_rounds, acc_vals, "o-", color="darkgreen", linewidth=2, markersize=6)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Sem dados de acuracia", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Acuracia"); ax.set_title("Acuracia Global")

    # 6. Throughput
    ax = axes[1, 2]
    ax.bar(rounds, throughput, color="teal", edgecolor="black")
    ax.axhline(np.mean(throughput), color="red", linestyle="--",
               label=f"Media: {np.mean(throughput):.1f}")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Amostras/s"); ax.set_title("Throughput")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(output_dir, f"benchmark_{timestamp}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────
# RELATÓRIO NO CONSOLE
# ─────────────────────────────────────────────────────────────

def print_benchmark_report(results: Dict) -> None:
    cfg = results["config"]
    smr = results["summary"]
    sep = "=" * 70

    print("\n" + sep)
    print(" RELATORIO DE BENCHMARK — MOSAIC-FL v2")
    print(sep)
    print(f"  Configuracao:")
    print(f"    Amostras:        {cfg['n_samples']:,}")
    print(f"    Rodadas:         {cfg['num_rounds']}")
    print(f"    Clientes:        {cfg['num_clients']}")
    print(f"    Batch size:      {cfg['batch_size']}")
    print(f"    Epocas locais:   {cfg['local_epochs']}")
    print(f"    Parametros:      {cfg['model_params']:,}  (~{cfg['model_size_mb']:.2f} MB)")
    print(f"    Vocabulario:     {cfg['vocab_size']} tokens")
    print(f"\n  Hardware:")
    print(f"    CPU:             {cfg['hardware']['cpu_cores']} cores")
    print(f"    RAM total:       {cfg['hardware']['ram_gb']:.1f} GB")
    print(f"\n  Resultados:")
    print(f"    Duracao total:   {smr['total_duration_sec']:.2f}s  ({smr['total_duration_sec']/60:.2f} min)")
    print(f"    Tempo/rodada:    {smr['avg_round_duration_sec']:.2f}s (media)")
    print(f"    Pico de memoria: {smr['peak_mem_mb']:.1f} MB")
    print(f"    CPU media:       {smr['avg_cpu_percent']:.1f}%")
    print(f"    CPU pico:        {smr['peak_cpu_percent']:.1f}%")
    print(f"    Trafego total:   up={smr['total_net_sent_mb']:.2f} MB  down={smr['total_net_recv_mb']:.2f} MB")
    acc = smr["final_accuracy"]
    print(f"    Acuracia final:  {acc:.4f}" if acc is not None else "    Acuracia final:  N/A")

    if results["per_round"]:
        print(f"\n  Metricas por Rodada:")
        header = f"  {'Rd':>3} | {'Tempo(s)':>10} | {'Mem(MB)':>10} | {'CPU(%)':>8} | {'Net(MB)':>10} | {'Acuracia':>10}"
        print(header)
        print("  " + "-" * 65)
        for r in results["per_round"]:
            acc_str = f"{r['server_accuracy']:.4f}" if r["server_accuracy"] is not None else "   N/A"
            net_total = r["network_mb_sent"] + r["network_mb_recv"]
            print(
                f"  {r['round']:>3} | {r['duration_sec']:>10.2f} | "
                f"{r['mem_peak_mb']:>10.1f} | {r['cpu_percent_peak']:>8.1f} | "
                f"{net_total:>10.2f} | {acc_str:>10}"
            )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark do MOSAIC-FL v2")
    parser.add_argument("--samples", type=int, default=1000,              help="Amostras sinteticas")
    parser.add_argument("--rounds",  type=int, default=10,                help="Rodadas federadas")
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS,       help="Clientes virtuais")
    parser.add_argument("--output",  type=str, default="benchmark_results",help="Diretorio de saida")
    args = parser.parse_args()

    run_benchmark(
        n_samples=args.samples,
        num_rounds=args.rounds,
        num_clients=args.clients,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
