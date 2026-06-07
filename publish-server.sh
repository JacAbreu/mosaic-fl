#!/bin/bash
# publish-server.sh
# Cria e publica o pacote mosaicfl-server no PyPI (ou servidor privado).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/infrastructure/server"

echo "Publicando mosaicfl-server..."
cd "$SERVER_DIR"

# Limpa builds antigos
rm -rf build dist *.egg-info

# Instala dependências de build
pip install --quiet build twine

# Build
python -m build

# Publica (descomente para PyPI real)
# twine upload dist/*

# Ou instala localmente para teste
pip install -e .

echo "mosaicfl-server publicado/instalado!"
echo ""
echo "Uso:"
echo "  mosaicfl-server --port 8080 --min-clients 3"