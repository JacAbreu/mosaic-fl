"""
Entrypoint do cliente MOSAIC-FL (mosaicfl.v2).

Uso:
    python -m mosaicfl_client --server 192.168.1.100:8080 --client-id hospital_a
    mosaicfl-client --server 192.168.1.100:8080 --client-id hospital_a
"""
from .runner import main

if __name__ == "__main__":
    main()
