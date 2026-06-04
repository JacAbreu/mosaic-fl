"""
client_daemon.py — entrypoint legado por script.

Delega para ProductionClient e main() em runner.py (fonte única de verdade).

Uso:
    python infrastructure/client/client_daemon.py --server HOST:8080 --client-id hospital_a

Variáveis de ambiente:
    FL_DATA_SOURCE=simulated|sgbd|csv
    FL_SERVER_ADDRESS, FL_CLIENT_ID, MOSAICFL_DB_URL, FL_CSV_PATH, ...
"""
import sys
from pathlib import Path

_INFRA_ROOT = Path(__file__).resolve().parent.parent
if str(_INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFRA_ROOT))

from client.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
