#!/bin/bash
# setup.sh - Mosaic-FL / MOSAICO-FL
# Configura ambiente virtual e instala dependências

set -e  # interrompe se qualquer comando falhar

VENV_DIR=".venv"
PYTHON_CMD="python3"

echo "========================================"
echo "  Mosaic-FL - Setup do Ambiente"
echo "========================================"

# Verifica se python3 está disponível
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Erro: python3 não encontrado. Instale Python >= 3.10."
    exit 1
fi

echo "[1/4] Python detectado: $($PYTHON_CMD --version)"

# Cria venv se não existir
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Ambiente virtual já existe em ./$VENV_DIR"
else
    echo "[2/4] Criando ambiente virtual em ./$VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Ativa venv
echo "[3/4] Ativando ambiente virtual..."
source $VENV_DIR/bin/activate

# Instala dependências
echo "[4/4] Instalando Mosaic-FL e dependências (incluindo dev)..."
pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "========================================"
echo "  Setup concluído com sucesso!"
echo "========================================"
echo ""
echo "Para ativar o ambiente futuramente:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Para executar os experimentos:"
echo "  python -m mosaicfl.experiments.runner"
echo "  ou"
echo "  python run.py"
echo ""
