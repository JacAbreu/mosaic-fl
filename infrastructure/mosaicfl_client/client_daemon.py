"""
client_daemon.py — entrypoint legado por script.

Delega para ProductionClient e main() em runner.py (fonte única de verdade).

Uso:
    python infrastructure/client/client_daemon.py --server HOST:8080 --client-id hospital_a

    # Ou com o pacote instalado:
    python -m infrastructure.client --server HOST:8080 --client-id hospital_a

Variáveis de ambiente:
    FL_DATA_SOURCE=simulated|sgbd|csv
    FL_SERVER_ADDRESS, FL_CLIENT_ID, MOSAICFL_DB_URL, FL_CSV_PATH, ...

Nota: Este script requer que o pacote mosaicfl esteja instalado ou que o
PYTHONPATH inclua o diretório raiz do projeto.
"""
from infrastructure.client.runner import main

if __name__ == "__main__":
    main()
