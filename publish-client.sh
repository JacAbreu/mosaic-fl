#!/bin/bash
# publish-client.sh
# Cria e publica o pacote mosaicfl-client no PyPI (ou servidor privado).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$SCRIPT_DIR/infrastructure/client"

echo "📦 Publicando mosaicfl-client..."
cd "$CLIENT_DIR"

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

echo "✅ mosaicfl-client publicado/instalado!"
echo ""
echo "Uso:"
echo "  mosaicfl-client --server 192.168.1.100:8080 --client-id hospital_a"