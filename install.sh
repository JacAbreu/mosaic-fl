#!/bin/bash
# install.sh
# Instala o ambiente completo de desenvolvimento do MOSAIC-FL.

set -e

echo "Instalando MOSAIC-FL (ambiente de desenvolvimento)..."

# 1. Cria venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Instala pacote core (src/)
pip install -e .

# 3. Instala server e client localmente (para desenvolvimento)
pip install -e infrastructure/mosaicfl_server
pip install -e infrastructure/mosaicfl_client
pip install -e infrastructure/mosaicfl_scheduler

# 4. Verifica instalação
echo ""
echo "Instalação completa!"
echo ""
echo "Comandos disponíveis:"
echo "  mosaicfl-server --port 8080 --min-clients 3"
echo "  mosaicfl-client --server localhost:8080 --client-id hospital_a"
echo ""
echo "Experimento TCC:"
echo "  python run_v2_unified.py"