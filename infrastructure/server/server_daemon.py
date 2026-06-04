"""
server_daemon.py — entrypoint legado por script.

Delega para FederatedServer e main() em runner.py (fonte única de verdade).
Mantido para compatibilidade com README e deploy manual:

    python infrastructure/server/server_daemon.py --port 8080 --min-clients 3

Equivalente a:

    python -m server.runner   # com infrastructure/ no PYTHONPATH
    mosaicfl-server           # após pip install mosaicfl-server
"""
import sys
from pathlib import Path

# Permite executar como arquivo: python infrastructure/server/server_daemon.py
_INFRA_ROOT = Path(__file__).resolve().parent.parent
if str(_INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFRA_ROOT))

from server.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
