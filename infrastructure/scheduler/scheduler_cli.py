"""
scheduler_cli.py
Entrypoint alternativo para o scheduler federado.

Delega para FederatedScheduler em scheduler_daemon.py.
Útil para:
  - Systemd service
  - Crontab
  - Docker ENTRYPOINT

Uso:
    python scheduler_cli.py --once
    python scheduler_cli.py --interval 1 --min-clients 2
"""
import sys
from pathlib import Path

# Garante que o diretório do scheduler está no path (ajuste conforme sua estrutura)
# Se estiver em src/schedule/:
# sys.path.insert(0, str(Path(__file__).parent))

from scheduler_daemon import FederatedScheduler, main

if __name__ == "__main__":
    main()