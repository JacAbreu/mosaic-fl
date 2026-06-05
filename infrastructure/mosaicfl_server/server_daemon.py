"""
server_daemon.py — entrypoint legado por script.

Delega para FederatedServer e main() em runner.py (fonte única de verdade).
Mantido para compatibilidade com README e deploy manual:

    python infrastructure/server/server_daemon.py --port 8080 --min-clients 3

Equivalente a:

    python -m infrastructure.server --port 8080 --min-clients 3
    mosaicfl-server           # após pip install mosaicfl-server

Nota: Este script requer que o pacote mosaicfl esteja instalado ou que o
PYTHONPATH inclua o diretório raiz do projeto.
"""
from infrastructure.server.runner import main

if __name__ == "__main__":
    main()
