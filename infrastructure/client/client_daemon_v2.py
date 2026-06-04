"""
client_daemon_v2.py — alias legado (mesmo comportamento que client_daemon.py).

Mantido para compatibilidade com documentação que cita client_daemon_v2.
Toda a lógica v2 (datasource + FedProxClient) está em runner.py.

Uso:
    FL_DATA_SOURCE=simulated python infrastructure/client/client_daemon_v2.py \\
        --server localhost:8080 --client-id hf1
"""
import sys
from pathlib import Path

_INFRA_ROOT = Path(__file__).resolve().parent.parent
if str(_INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFRA_ROOT))

from client.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
