"""resources.py — Amostragem best-effort de recursos computacionais (GPU).

Compartilhado entre o loop manual (Caminho A, experiments/training/core/fl_core/
manual_loop.py) e o FedProxClient (Caminho B, client.py) — extraído para módulo
único para evitar duas implementações divergentes da mesma amostragem via nvidia-smi.
"""
import subprocess
from typing import Optional


def sample_gpu_power_w() -> Optional[float]:
    """Amostra a potência instantânea da GPU (Watts) via nvidia-smi.

    Retorna None em qualquer falha (sem GPU NVIDIA, driver ausente, timeout) —
    nunca deve interromper o treinamento por causa de coleta de métrica de energia.
    Relevante para viabilidade de implantação em ambientes com energia/água
    limitadas para resfriamento — custo energético real, não estimado.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
