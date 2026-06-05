# Makefile for MOSAIC-FL project
# Federated Learning for Clinical Prediction

.PHONY: help venv setup install run clean test test-cov lint fmt pre-commit docker-build docker-up docker-down

# Detect Python environment
VENV_DIR := .venv

ifdef VIRTUAL_ENV
	PYTHON      := python
	PIP         := pip
	PYTHON_VENV := python
	IN_VENV     := yes
else
	ifneq ($(wildcard $(VENV_DIR)/bin/python),)
		PYTHON      := $(VENV_DIR)/bin/python
		PIP         := $(VENV_DIR)/bin/pip
		PYTHON_VENV := $(VENV_DIR)/bin/python
		IN_VENV     := venv_exists
	else
		PYTHON      := python3
		PIP         := pip3
		PYTHON_VENV := python3
		IN_VENV     := no
	endif
endif

# Default target
help:
	@echo "MOSAIC-FL - Comandos disponiveis"
	@echo ""
	@echo "  SETUP"
	@echo "    make venv        - Cria ambiente virtual (.venv)"
	@echo "    make setup       - Cria venv e instala dependencias"
	@echo "    make install     - Instala o pacote em modo editavel"
	@echo ""
	@echo "  EXECUCAO"
	@echo "    make run         - Executa experimentos"
	@echo "    make test        - Executa testes"
	@echo "    make test-cov    - Testes com relatorio de cobertura"
	@echo ""
	@echo "  QUALIDADE"
	@echo "    make lint        - Verifica estilo com ruff"
	@echo "    make fmt         - Formata codigo com ruff"
	@echo "    make pre-commit  - Instala hooks de pre-commit"
	@echo ""
	@echo "  MANUTENCAO"
	@echo "    make clean       - Remove arquivos temporarios"
	@echo ""
	@echo "  DOCKER"
	@echo "    make docker-up   - Inicia servicos Docker"
	@echo "    make docker-down - Para servicos Docker"
	@echo ""
	@echo "  Ambiente: $(IN_VENV)"

# Create virtual environment
venv:
ifeq ($(wildcard $(VENV_DIR)/bin/python),)
	python3 -m venv $(VENV_DIR)
	@echo "Ambiente virtual criado em $(VENV_DIR)/"
	@echo "Para ativar: source $(VENV_DIR)/bin/activate"
else
	@echo "Ambiente virtual ja existe em $(VENV_DIR)/"
endif

# Setup: create venv and install all dependencies
setup:
	@$(MAKE) venv
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install -e .
	$(VENV_DIR)/bin/pip install pytest pytest-cov ruff pre-commit
	$(VENV_DIR)/bin/pre-commit install
	@echo ""
	@echo "Setup concluido."
	@echo "  Proximos passos:"
	@echo "    source $(VENV_DIR)/bin/activate"
	@echo "    make test"

# Install package in editable mode
install:
	$(PIP) install -e .

# Run experiments
run:
	$(PYTHON_VENV) -m mosaicfl.experiments.runner

# Run all tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short

# Run tests with coverage report
test-cov:
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short --cov=src/mosaicfl --cov-report=term-missing

# Lint with ruff (check only)
lint:
	$(VENV_DIR)/bin/ruff check src/ tests/ benchmark.py

# Format with ruff
fmt:
	$(VENV_DIR)/bin/ruff format src/ tests/ benchmark.py

# Install pre-commit hooks
pre-commit:
	$(VENV_DIR)/bin/pre-commit install
	@echo "Hooks instalados. Serao executados automaticamente em cada commit."

# Clean build artifacts and caches
clean:
	rm -rf $(VENV_DIR) __pycache__ .pytest_cache .coverage htmlcov dist build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Limpeza concluida."

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
