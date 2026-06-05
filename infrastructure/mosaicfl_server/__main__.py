"""
Entrypoint do servidor MOSAIC-FL (mosaicfl.v2).

Uso:
    python -m mosaicfl_server --port 8080 --min-clients 3
    mosaicfl-server --port 8080 --min-clients 3
"""
from .runner import main

if __name__ == "__main__":
    main()
