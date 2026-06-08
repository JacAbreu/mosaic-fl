#!/bin/bash
# setup.sh — Configura o ambiente de desenvolvimento do MOSAIC-FL.
#
# O que este script faz:
#   1. Verifica se Python >= 3.10 esta disponivel
#   2. Cria o ambiente virtual .venv (se ainda nao existir)
#   3. Instala o pacote principal com extras de desenvolvimento (pytest, ruff, etc.)
#   4. Instala os subpacotes de infraestrutura (server, client, scheduler)
#
# Uso:
#   bash setup.sh
#   ou: make setup

set -e

VENV_DIR=".venv"
PYTHON_CMD="python3"

echo "========================================"
echo "  MOSAIC-FL - Configuracao do Ambiente"
echo "========================================"

# Verifica python3
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Erro: python3 nao encontrado. Instale Python >= 3.10."
    exit 1
fi

echo "[1/4] Python detectado: $($PYTHON_CMD --version)"

# Cria venv se nao existir
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Ambiente virtual ja existe em ./$VENV_DIR"
else
    echo "[2/4] Criando ambiente virtual em ./$VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Ativa venv
echo "[3/4] Ativando ambiente virtual..."
source $VENV_DIR/bin/activate

# Instala dependencias
echo "[4/4] Instalando MOSAIC-FL e dependencias..."
pip install --upgrade pip --quiet

# Pacote principal + extras de dev (pytest, ruff, etc.)
pip install -e ".[dev]"

# Subpacotes de infraestrutura
pip install -e infrastructure/mosaicfl_server
pip install -e infrastructure/mosaicfl_client
pip install -e infrastructure/mosaicfl_scheduler

echo ""
echo "========================================"
echo "  Ambiente configurado com sucesso!"
echo "========================================"
echo ""
echo "Para ativar o ambiente em outros terminais:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Comandos disponiveis:"
echo "  make test              # testes unitarios"
echo "  make experiment        # roda os experimentos"
echo "  make sim               # simulacao FL local (sem rede)"
echo "  make superlink         # inicia servidor FL (requer TLS)"
echo "  make supernode         # inicia cliente FL (requer TLS)"
echo ""
echo "Para gerar certificados TLS (primeira vez em rede real):"
echo "  bash scripts/gerar_certs_tls.sh"
echo ""
