"""
client_daemon_v2.py — alias legado (mesmo comportamento que client_daemon.py).

Mantido para compatibilidade com documentação que cita client_daemon_v2.
Toda a lógica v2 (datasource + FedProxClient) está em runner.py.

Uso:
    FL_DATA_SOURCE=simulated python infrastructure/client/client_daemon_v2.py \\
        --server localhost:8080 --client-id hf1

    # Ou com o pacote instalado:
    python -m infrastructure.client --server localhost:8080 --client-id hf1

Nota: Este script requer que o pacote mosaicfl esteja instalado ou que o
PYTHONPATH inclua o diretório raiz do projeto.

TODO: Consolidar com client_daemon.py em versão futura para evitar
duplicação de entrypoints.
"""
from infrastructure.client.runner import main

if __name__ == "__main__":
    main()
