#!/bin/bash
# setup.sh — Configura o ambiente de desenvolvimento do MOSAIC-FL.
#
# O que este script faz:
#   1. Verifica se Python >= 3.10 esta disponivel
#   2. Cria o ambiente virtual .venv (se ainda nao existir)
#   3. Instala o pacote principal com extras de desenvolvimento (pytest, ruff, etc.)
#   4. Instala os subpacotes de infraestrutura (server, client, scheduler)
#   5. Instala o Ollama (backend LLM do RAG)
#   6. Faz o pull do modelo LLM configurado (FL_LLM_MODEL, padrao: gemma3:4b)
#
# Uso:
#   bash setup.sh
#   ou: make setup
#
# Variavel de ambiente opcional:
#   FL_LLM_MODEL=llama3.2:3b bash setup.sh   # troca o modelo LLM

set -e

VENV_DIR=".venv"
PYTHON_CMD="python3"
LLM_MODEL="${FL_LLM_MODEL:-gemma3:4b}"

echo "========================================"
echo "  MOSAIC-FL - Configuracao do Ambiente"
echo "========================================"

# Verifica python3
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Erro: python3 nao encontrado. Instale Python >= 3.10."
    exit 1
fi

echo "[1/6] Python detectado: $($PYTHON_CMD --version)"

# Cria venv se nao existir
if [ -d "$VENV_DIR" ]; then
    echo "[2/6] Ambiente virtual ja existe em ./$VENV_DIR"
else
    echo "[2/6] Criando ambiente virtual em ./$VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Ativa venv
echo "[3/6] Ativando ambiente virtual..."
source $VENV_DIR/bin/activate

# Instala dependencias
echo "[4/6] Instalando MOSAIC-FL e dependencias..."
pip install --upgrade pip --quiet

# Pacote principal + extras de dev (pytest, ruff, etc.)
pip install -e ".[dev]"

# Subpacotes de infraestrutura
pip install -e infrastructure/mosaicfl_server
pip install -e infrastructure/mosaicfl_client
pip install -e infrastructure/mosaicfl_scheduler

# Instala Ollama (backend LLM do RAG)
echo "[5/6] Verificando Ollama..."
if command -v ollama &> /dev/null; then
    echo "       Ollama ja instalado: $(ollama --version)"
else
    echo "       Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "       Ollama instalado: $(ollama --version)"
fi

# Pull do modelo LLM (gemma3:4b por padrao)
echo "[6/6] Verificando modelo LLM: $LLM_MODEL"
# Garante que o servidor Ollama esta rodando para o pull
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "       Iniciando ollama serve em background..."
    ollama serve > /tmp/ollama_setup.log 2>&1 &
    OLLAMA_PID=$!
    sleep 3
fi

if ollama list 2>/dev/null | grep -q "^${LLM_MODEL}"; then
    echo "       Modelo $LLM_MODEL ja disponivel"
else
    echo "       Baixando $LLM_MODEL (pode levar alguns minutos)..."
    ollama pull "$LLM_MODEL"
fi

echo ""
echo "========================================"
echo "  Ambiente configurado com sucesso!"
echo "========================================"
echo ""
echo "Para ativar o ambiente em outros terminais:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Antes de treinar, inicie o Ollama:"
echo "  ollama serve &"
echo ""
echo "Comandos disponiveis:"
echo "  make test              # testes unitarios"
echo "  make experiment        # roda os experimentos"
echo "  make sim               # simulacao FL local (sem rede)"
echo "  make training-full     # pipeline completo (4 fases + RAG)"
echo "  make ollama-check      # valida se Ollama esta pronto"
echo "  make superlink         # inicia servidor FL (requer TLS)"
echo "  make supernode         # inicia cliente FL (requer TLS)"
echo ""
echo "Para gerar certificados TLS (primeira vez em rede real):"
echo "  bash scripts/gerar_certs_tls.sh"
echo ""
