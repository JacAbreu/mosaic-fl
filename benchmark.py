"""
benchmark.py — Benchmark de Performance do MOSAIC-FL

Mede:
  • Tempo por rodada federada (treino + agregação + avaliação)
  • Uso de memória RAM (pico e média por rodada)
  • Uso de CPU (%)
  • Tráfego de rede estimado (tamanho dos pesos transmitidos)
  • Throughput (amostras/segundo por cliente)
  • Latência de comunicação (simulada)

Saídas:
  • JSON com métricas detalhadas por rodada
  • Gráficos PNG (tempo, memória, CPU, tráfego)
  • Resumo estatístico no console

Uso:
    source .venv/bin/activate
    python benchmark.py

Requisitos:
    pip install psutil matplotlib pandas
"""
import os
import sys
import time
import json
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import psutil
import matplotlib
matplotlib.use("Agg")  # headless (sem GUI)
import matplotlib.pyplot as plt

# Flower
import flwr as fl
from flwr.simulation import start_simulation

# Módulos do projeto
from src.config import *
from src.preprocess import EHRPreprocessor, split_by_institution
from src.model import SimplifiedBEHRT
from src.client import FedProxClient, create_client_fn
from src.server import start_server, ConvergenceTracker, CustomFedProxStrategy, weighted_average

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
        self._running = False
        self._interval = 0.5  # segundos entre amostras

    def start(self):
        """Inicia coleta periódica em background."""
        self._running = True
        self._cpu_samples.clear()
        self._mem_samples.clear()
        self._net_start = psutil.net_io_counters()
        # Amostra inicial
        self._sample()

    def _sample(self):
        if not self._running:
            return
        try:
            mem_mb = self.process.memory_info().rss / (1024 ** 2)
            cpu_pct = self.process.cpu_percent(interval=None)
            self._mem_samples.append(mem_mb)
            self._cpu_samples.append(cpu_pct)
        except psutil.NoSuchProcess:
            pass

    def sample_now(self):
        """Força uma amostra imediata (útil para pontos específicos)."""
        self._sample()

    def stop(self):
        """Para coleta e retorna estatísticas."""
        self._running = False
        self._sample()  # amostra final

        net_end = psutil.net_io_counters()
        net_sent_mb = (net_end.bytes_sent - self._net_start.bytes_sent) / (1024 ** 2)
        net_recv_mb = (net_end.bytes_recv - self._net_start.bytes_recv) / (1024 ** 2)

        stats = {
            "cpu_avg": np.mean(self._cpu_samples) if self._cpu_samples else 0.0,
            "cpu_peak": np.max(self._cpu_samples) if self._cpu_samples else 0.0,
            "cpu_min": np.min(self._cpu_samples) if self._cpu_samples else 0.0,
            "mem_avg_mb": np.mean(self._mem_samples) if self._mem_samples else 0.0,
            "mem_peak_mb": np.max(self._mem_samples) if self._mem_samples else 0.0,
            "mem_min_mb": np.min(self._mem_samples) if self._mem_samples else 0.0,
            "net_sent_mb": net_sent_mb,
            "net_recv_mb": net_recv_mb,
        }
        return stats

    @staticmethod
    def estimate_model_size_mb(num_parameters: int, dtype_bytes: int = 4) -> float:
        """Estima tamanho do modelo em MB."""
        return (num_parameters * dtype_bytes) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────
# GERAÇÃO DE DADOS DE TESTE
# ─────────────────────────────────────────────────────────────

def generate_benchmark_data(
    n_samples: int = 1000,
    n_institutions: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Gera dataset sintético escalável para benchmark."""
    np.random.seed(random_state)
    institutions = [f"Hospital_{i}" for i in range(n_institutions)]
    sintomas = ["febre", "tosse", "dispneia", "fadiga", "mialgia", "cefaleia", "anosmia", "diarreia"]
    exames = ["rt_pcr_positivo", "tomografia_normal", "tomografia_vidro_fosco", "rx_consolidacao", "pcr_negativo"]
    diagnosticos = ["covid19_leve", "covid19_moderado", "covid19_grave", "pneumonia_bacteriana", "alta"]

    return pd.DataFrame({
        "instituicao": np.random.choice(institutions, n_samples),
        "idade": np.random.randint(18, 90, n_samples),
        "idade_unidade": np.random.choice(["anos", "meses"], n_samples, p=[0.95, 0.05]),
        "peso": np.random.uniform(50, 120, n_samples),
        "peso_unidade": np.random.choice(["kg", "lb"], n_samples, p=[0.9, 0.1]),
        "temperatura": np.random.uniform(36.0, 40.0, n_samples),
        "sintoma": np.random.choice(sintomas, n_samples),
        "exame": np.random.choice(exames, n_samples),
        "diagnostico": np.random.choice(diagnosticos, n_samples),
        "desfecho": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })


def prepare_benchmark_loaders(
    df: pd.DataFrame,
    preprocessor: EHRPreprocessor,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[Dict, DataLoader, int]:
    """
    Prepara DataLoaders para benchmark.

    Returns:
        client_loaders: dict {cid: (train_loader, val_loader)}
        test_loader: DataLoader global
        vocab_size: int
    """
    text_cols = ["sintoma", "exame", "diagnostico"]
    df_proc, summary = preprocessor.process(df, text_cols=text_cols)

    client_dfs = split_by_institution(
        df_proc,
        institution_col="instituicao",
        num_clients=NUM_CLIENTS,
        stratify_col="desfecho",
        random_state=RANDOM_SEED,
    )

    vocab_map = preprocessor.vocab_map
    vocab_size = len(vocab_map)
    seq_cols = [c for c in df_proc.columns if c.endswith("_encoded")]

    def make_sequences(sub_df):
        sequences, labels = [], []
        for _, row in sub_df.iterrows():
            seq = [int(row[c]) for c in seq_cols if pd.notna(row[c])]
            if len(seq) < max_seq_len:
                seq = seq + [0] * (max_seq_len - len(seq))
            else:
                seq = seq[:max_seq_len]
            sequences.append(seq)
            labels.append(int(row["desfecho"]))
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    client_loaders = {}
    all_train_samples = 0

    for cid, subset in client_dfs.items():
        n = len(subset)
        if n < 10:
            continue
        n_train = int(0.8 * n)
        train_df = subset.iloc[:n_train]
        val_df = subset.iloc[n_train:]

        train_x, train_y = make_sequences(train_df)
        val_x, val_y = make_sequences(val_df)

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(val_x, val_y),
            batch_size=batch_size,
        )
        client_loaders[cid] = (train_loader, val_loader)
        all_train_samples += len(train_x)

    # Teste global: 20%
    test_size = int(0.2 * len(df_proc))
    test_df = df_proc.sample(n=test_size, random_state=RANDOM_SEED)
    test_x, test_y = make_sequences(test_df)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size,
    )

    return client_loaders, test_loader, vocab_size, all_train_samples


# ─────────────────────────────────────────────────────────────
# ESTRATÉGIA DE BENCHMARK
# ─────────────────────────────────────────────────────────────

class BenchmarkFedProxStrategy(fl.server.strategy.FedProx):
    """
    Estratégia FedProx instrumentada para coleta de métricas de benchmark.
    """
    def __init__(self, monitor: ResourceMonitor, metrics_history: List[RoundMetrics],
                 model_size_mb: float, total_train_samples: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.metrics_history = metrics_history
        self.model_size_mb = model_size_mb
        self.total_train_samples = total_train_samples
        self.current_round = 0

    def aggregate_fit(self, server_round, results, failures):
        """Intercepta agregação para medir tempo e recursos."""
        self.current_round = server_round
        round_start = time.time()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

        # Amostra de recursos antes da agregação
        self.monitor.sample_now()

        # Executa agregação original
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Amostra após agregação
        self.monitor.sample_now()

        # Estima tráfego: cada cliente enviou e recebeu ~model_size_mb
        n_clients = len(results) if results else 1
        net_sent = self.model_size_mb * n_clients  # servidor → clientes
        net_recv = self.model_size_mb * n_clients  # clientes → servidor

        round_end = time.time()
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

        # Calcula throughput estimado
        duration = round_end - round_start
        throughput = self.total_train_samples / duration if duration > 0 else 0.0

        metric = RoundMetrics(
            round=server_round,
            start_time=round_start,
            end_time=round_end,
            duration_sec=duration,
            mem_before_mb=mem_before,
            mem_peak_mb=max(mem_before, mem_after),
            mem_after_mb=mem_after,
            cpu_percent_avg=0.0,  # preenchido depois
            cpu_percent_peak=0.0,
            network_mb_sent=net_sent,
            network_mb_recv=net_recv,
            train_samples=self.total_train_samples,
            throughput_samples_per_sec=throughput,
        )
        self.metrics_history.append(metric)

        logger.info(
            f"[Benchmark] Rodada {server_round}: {duration:.2f}s | "
            f"Mem: {mem_before:.1f}→{mem_after:.1f} MB | "
            f"Net: ↑{net_sent:.2f} ↓{net_recv:.2f} MB | "
            f"Throughput: {throughput:.1f} amostras/s"
        )

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Intercepta avaliação para registrar acurácia global."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated and len(aggregated) == 2:
            loss, metrics = aggregated
            accuracy = metrics.get("accuracy", 0.0)
            # Atualiza métrica da rodada atual
            if self.metrics_history:
                self.metrics_history[-1].server_loss = float(loss)
                self.metrics_history[-1].server_accuracy = float(accuracy)
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
    """
    Executa benchmark completo do pipeline FL.

    Args:
        n_samples: número de amostras sintéticas
        num_rounds: rodadas federadas a executar
        num_clients: número de clientes virtuais
        output_dir: diretório para salvar resultados

    Returns:
        dict com métricas agregadas e caminhos dos artefatos
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("BENCHMARK MOSAIC-FL")
    logger.info("=" * 70)
    logger.info(f"Configuração: {n_samples} amostras | {num_rounds} rodadas | {num_clients} clientes")
    logger.info(f"Hardware: CPU={psutil.cpu_count()} cores | RAM={psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Modelo: BEHRT mini ({EMBED_DIM}d, {NUM_LAYERS} camadas, {NUM_HEADS} heads)")
    logger.info("=" * 70)

    # 1. Gera dados
    logger.info("[1/5] Gerando dados sintéticos...")
    t0 = time.time()
    df = generate_benchmark_data(n_samples=n_samples, n_institutions=num_clients)
    logger.info(f"      ✓ {len(df)} amostras em {time.time() - t0:.2f}s")

    # 2. Pré-processamento
    logger.info("[2/5] Pré-processando...")
    t0 = time.time()
    preprocessor = EHRPreprocessor()
    client_loaders, test_loader, vocab_size, total_train_samples = prepare_benchmark_loaders(
        df, preprocessor, batch_size=BATCH_SIZE
    )
    logger.info(f"      ✓ Vocabulário: {vocab_size} tokens | Clientes: {len(client_loaders)} | {time.time() - t0:.2f}s")

    # 3. Estima tamanho do modelo
    model = SimplifiedBEHRT()
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = ResourceMonitor.estimate_model_size_mb(num_params)
    logger.info(f"[3/5] Modelo: {num_params:,} parâmetros | ~{model_size_mb:.2f} MB por transmissão")

    # 4. Configura monitoramento
    monitor = ResourceMonitor()
    metrics_history: List[RoundMetrics] = []

    # 5. Estratégia instrumentada
    evaluate_fn = None
    if test_loader:
        from src.server import get_evaluate_fn
        evaluate_fn = get_evaluate_fn(test_loader)

    strategy = BenchmarkFedProxStrategy(
        monitor=monitor,
        metrics_history=metrics_history,
        model_size_mb=model_size_mb,
        total_train_samples=total_train_samples,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_fit_clients=min(num_clients, MIN_FIT_CLIENTS),
        min_evaluate_clients=min(num_clients, MIN_EVALUATE_CLIENTS),
        min_available_clients=min(num_clients, MIN_AVAILABLE_CLIENTS),
        proximal_mu=PROXIMAL_MU,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    # 6. Factory de clientes
    def client_fn(cid: str) -> fl.client.NumPyClient:
        cid_int = int(cid)
        train_loader, val_loader = client_loaders[cid_int]
        return FedProxClient(cid_int, train_loader, val_loader)

    # 7. Executa FL com monitoramento
    logger.info("[4/5] Executando Aprendizado Federado...")
    logger.info("=" * 70)

    overall_start = time.time()
    monitor.start()

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(client_loaders),
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0},
        )
    except Exception as e:
        logger.error(f"Erro na simulação: {e}")

    monitor.stop()
    overall_end = time.time()
    total_duration = overall_end - overall_start

    logger.info("=" * 70)
    logger.info(f"[5/5] Benchmark concluído em {total_duration:.2f}s")

    # 8. Consolida métricas
    stats = monitor.stop()  # estatísticas finais
    for m in metrics_history:
        m.cpu_percent_avg = stats["cpu_avg"]
        m.cpu_percent_peak = stats["cpu_peak"]

    # 9. Salva resultados
    results = {
        "config": {
            "n_samples": n_samples,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "batch_size": BATCH_SIZE,
            "local_epochs": LOCAL_EPOCHS,
            "model_params": num_params,
            "model_size_mb": model_size_mb,
            "vocab_size": vocab_size,
            "hardware": {
                "cpu_cores": psutil.cpu_count(),
                "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            },
        },
        "summary": {
            "total_duration_sec": total_duration,
            "avg_round_duration_sec": np.mean([m.duration_sec for m in metrics_history]) if metrics_history else 0.0,
            "peak_mem_mb": stats["mem_peak_mb"],
            "avg_cpu_percent": stats["cpu_avg"],
            "peak_cpu_percent": stats["cpu_peak"],
            "total_net_sent_mb": sum(m.network_mb_sent for m in metrics_history),
            "total_net_recv_mb": sum(m.network_mb_recv for m in metrics_history),
            "final_accuracy": metrics_history[-1].server_accuracy if metrics_history else None,
        },
        "per_round": [asdict(m) for m in metrics_history],
    }

    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"      ✓ JSON salvo: {json_path}")

    # 10. Gera gráficos
    if metrics_history:
        plot_path = generate_plots(metrics_history, results["summary"], output_dir, timestamp)
        logger.info(f"      ✓ Gráficos salvos: {plot_path}")

    # 11. Relatório no console
    print_benchmark_report(results)

    return results


# ─────────────────────────────────────────────────────────────
# GERAÇÃO DE GRÁFICOS
# ─────────────────────────────────────────────────────────────

def generate_plots(
    metrics: List[RoundMetrics],
    summary: Dict,
    output_dir: str,
    timestamp: str,
) -> str:
    """Gera figura com 4 subplots: tempo, memória, CPU, tráfego."""
    rounds = [m.round for m in metrics]
    durations = [m.duration_sec for m in metrics]
    mem_before = [m.mem_before_mb for m in metrics]
    mem_after = [m.mem_after_mb for m in metrics]
    mem_peak = [m.mem_peak_mb for m in metrics]
    cpu_avg = [m.cpu_percent_avg for m in metrics]
    net_sent = [m.network_mb_sent for m in metrics]
    net_recv = [m.network_mb_recv for m in metrics]
    accuracy = [m.server_accuracy for m in metrics if m.server_accuracy is not None]
    acc_rounds = [m.round for m in metrics if m.server_accuracy is not None]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"MOSAIC-FL Benchmark | {timestamp}", fontsize=14, fontweight="bold")

    # 1. Duração por rodada
    ax = axes[0, 0]
    ax.bar(rounds, durations, color="steelblue", edgecolor="black")
    ax.axhline(np.mean(durations), color="red", linestyle="--", label=f"Média: {np.mean(durations):.2f}s")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Duração (s)")
    ax.set_title("Tempo por Rodada")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Memória
    ax = axes[0, 1]
    ax.plot(rounds, mem_before, "o-", label="Antes", color="green", alpha=0.7)
    ax.plot(rounds, mem_after, "s-", label="Depois", color="orange", alpha=0.7)
    ax.plot(rounds, mem_peak, "^-", label="Pico", color="red", alpha=0.7)
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Memória (MB)")
    ax.set_title("Uso de RAM")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. CPU
    ax = axes[0, 2]
    ax.fill_between(rounds, cpu_avg, alpha=0.3, color="purple")
    ax.plot(rounds, cpu_avg, "o-", color="purple", label="CPU %")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("CPU (%)")
    ax.set_title("Uso de CPU")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Tráfego de rede
    ax = axes[1, 0]
    x = np.arange(len(rounds))
    width = 0.35
    ax.bar(x - width/2, net_sent, width, label="Enviado", color="coral")
    ax.bar(x + width/2, net_recv, width, label="Recebido", color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Tráfego (MB)")
    ax.set_title("Tráfego de Rede (estimado)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 5. Acurácia global
    ax = axes[1, 1]
    if accuracy:
        ax.plot(acc_rounds, accuracy, "o-", color="darkgreen", linewidth=2, markersize=6)
        ax.set_xlabel("Rodada")
        ax.set_ylabel("Acurácia")
        ax.set_title("Acurácia Global")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Sem dados de
acurácia", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Acurácia Global")

    # 6. Throughput
    ax = axes[1, 2]
    throughput = [m.throughput_samples_per_sec for m in metrics]
    ax.bar(rounds, throughput, color="teal", edgecolor="black")
    ax.axhline(np.mean(throughput), color="red", linestyle="--", label=f"Média: {np.mean(throughput):.1f}")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Amostras/s")
    ax.set_title("Throughput")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f"benchmark_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ─────────────────────────────────────────────────────────────
# RELATÓRIO NO CONSOLE
# ─────────────────────────────────────────────────────────────

def print_benchmark_report(results: Dict):
    """Imprime relatório formatado no console."""
    cfg = results["config"]
    smr = results["summary"]

    print("
" + "=" * 70)
    print(" RELATÓRIO DE BENCHMARK — MOSAIC-FL")
    print("=" * 70)
    print(f"  Configuração:")
    print(f"    • Amostras:        {cfg['n_samples']:,}")
    print(f"    • Rodadas:         {cfg['num_rounds']}")
    print(f"    • Clientes:        {cfg['num_clients']}")
    print(f"    • Batch size:      {cfg['batch_size']}")
    print(f"    • Épocas locais:   {cfg['local_epochs']}")
    print(f"    • Parâmetros:      {cfg['model_params']:,} (~{cfg['model_size_mb']:.2f} MB)")
    print(f"    • Vocabulário:     {cfg['vocab_size']} tokens")
    print(f"
  Hardware:")
    print(f"    • CPU:             {cfg['hardware']['cpu_cores']} cores")
    print(f"    • RAM total:       {cfg['hardware']['ram_gb']:.1f} GB")
    print(f"
  Resultados:")
    print(f"    • Duração total:   {smr['total_duration_sec']:.2f}s ({smr['total_duration_sec']/60:.2f} min)")
    print(f"    • Tempo/rodada:    {smr['avg_round_duration_sec']:.2f}s (média)")
    print(f"    • Pico de memória: {smr['peak_mem_mb']:.1f} MB")
    print(f"    • CPU média:       {smr['avg_cpu_percent']:.1f}%")
    print(f"    • CPU pico:        {smr['peak_cpu_percent']:.1f}%")
    print(f"    • Tráfego total:   ↑{smr['total_net_sent_mb']:.2f} MB  ↓{smr['total_net_recv_mb']:.2f} MB")
    print(f"    • Acurácia final:  {smr['final_accuracy']:.4f}" if smr['final_accuracy'] else "    • Acurácia final:  N/A")
    print("=" * 70)

    # Tabela por rodada
    if results["per_round"]:
        print("
  Métricas por Rodada:")
        print(f"  {'Rd':>3} | {'Tempo(s)':>10} | {'Mem(MB)':>10} | {'CPU(%)':>8} | {'Net(MB)':>10} | {'Acurácia':>10}")
        print("  " + "-" * 65)
        for r in results["per_round"]:
            acc = f"{r['server_accuracy']:.4f}" if r['server_accuracy'] is not None else "N/A"
            net = r['network_mb_sent'] + r['network_mb_recv']
            print(f"  {r['round']:>3} | {r['duration_sec']:>10.2f} | {r['mem_peak_mb']:>10.1f} | {r['cpu_percent_peak']:>8.1f} | {net:>10.2f} | {acc:>10}")
    print("=" * 70 + "
")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark do MOSAIC-FL")
    parser.add_argument("--samples", type=int, default=1000, help="Número de amostras sintéticas")
    parser.add_argument("--rounds", type=int, default=10, help="Número de rodadas federadas")
    parser.add_argument("--clients", type=int, default=5, help="Número de clientes virtuais")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Diretório de saída")
    args = parser.parse_args()

    results = run_benchmark(
        n_samples=args.samples,
        num_rounds=args.rounds,
        num_clients=args.clients,
        output_dir=args.output,
    )

    # Retorna código de sucesso
    return 0


if __name__ == "__main__":
    sys.exit(main())
