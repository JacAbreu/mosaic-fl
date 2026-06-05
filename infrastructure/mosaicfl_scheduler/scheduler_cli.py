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

    # Ou com o pacote instalado:
    python -m infrastructure.scheduler --once

Nota: Este script requer que o pacote mosaicfl esteja instalado ou que o
PYTHONPATH inclua o diretório raiz do projeto.
"""
from infrastructure.scheduler.scheduler_daemon import FederatedScheduler, main

if __name__ == "__main__":
    main()