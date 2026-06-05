# Makefile for MOSAIC-FL project
# Federal Learning for Clinical Prediction

.PHONY: help venv setup install run clean test test-cov lint format docker-build docker-up docker-down

# Detect Python environment
VENV_DIR := .venv

# Check if running inside virtual environment
ifdef VIRTUAL_ENV
	# Venv is already activated
	PYTHON := python
	PIP := pip
	PYTHON_VENV := python
	IN_VENV := yes
else
	# Check if .venv exists
	ifneq ($(wildcard $(VENV_DIR)/bin/python),)
		PYTHON := $(VENV_DIR)/bin/python
		PIP := $(VENV_DIR)/bin/pip
		PYTHON_VENV := $(VENV_DIR)/bin/python
		IN_VENV := venv_exists
	else
		PYTHON := python3
		PIP := pip3
		PYTHON_VENV := python3
		IN_VENV := no
	endif
endif

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║           MOSAIC-FL - Comandos Disponíveis                 ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 SETUP:"
	@echo "  make venv        - Cria ambiente virtual (.venv)"
	@echo "  make setup       - Cria venv e instala dependências"
	@echo "  make install     - Instala o pacote em modo editável"
	@echo ""
	@echo "🚀 EXECUÇÃO:"
	@echo "  make run         - Executa experimentos (runner)"
	@echo "  make test        - Executa testes unitários"
	@echo "  make test-cov    - Testes com cobertura"
	@echo ""
	@echo "🧹 MANUTENÇÃO:"
	@echo "  make clean       - Remove arquivos temporários"
	@echo ""
	@echo "🐳 DOCKER:"
	@echo "  make docker-up   - Inicia serviços Docker"
	@echo "  make docker-down - Para serviços Docker"
	@echo ""
	@echo "💡 Status do ambiente: $(IN_VENV)"

# Create virtual environment (only if not exists)
venv:
ifeq ($(wildcard $(VENV_DIR)/bin/python),)
	@echo "📦 Criando ambiente virtual..."
	python3 -m venv $(VENV_DIR)
	@echo "✅ Ambiente virtual criado em $(VENV_DIR)/"
	@echo "💡 Execute: source $(VENV_DIR)/bin/activate"
else
	@echo "✅ Ambiente virtual já existe em $(VENV_DIR)/"
	@echo "💡 Para ativar: source $(VENV_DIR)/bin/activate"
endif

# Setup: create venv and install dependencies
setup:
	@echo "🔧 Iniciando setup do MOSAIC-FL..."
	@$(MAKE) venv
	@echo "📥 Instalando dependências..."
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install -e .
	$(VENV_DIR)/bin/pip install pytest pytest-cov black ruff
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ SETUP COMPLETO!                                       ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Próximos passos:"
	@echo "   1. Ative o ambiente: source $(VENV_DIR)/bin/activate"
	@echo "   2. Execute testes: make test"
	@echo "   3. Rode experimentos: make run"
	@echo ""

# Install package in editable mode
install:
	$(PIP) install -e .

# Run experiments
run:
	$(PYTHON_VENV) -m mosaicfl.experiments.runner

# Run all tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short --cov=src/mosaicfl --cov-report=term-missing

# Clean build artifacts and cache
clean:
	@echo "Limpando arquivos temporários..."
	rm -rf $(VENV_DIR) __pycache__ .pytest_cache .coverage htmlcov dist build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Limpeza completa!"

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f