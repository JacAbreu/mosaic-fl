"""
Entrypoint do scheduler MOSAIC-FL.

Uso:
    python -m mosaicfl_scheduler
    # ou após instalação via pip:
    mosaicfl-scheduler --interval 6 --min-clients 3
"""
from .scheduler_daemon import main

if __name__ == "__main__":
    main()
