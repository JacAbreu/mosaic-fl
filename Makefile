PYTHON   := .venv/bin/python
PYTEST   := $(PYTHON) -m pytest
FLWR     := .venv/bin/flwr

# Configurações do SuperLink (sobrescrevíveis por variável de ambiente)
FL_TLS_CERT_DIR      ?= certs
FL_SUPERLINK_ADDRESS ?= localhost:9091
FL_CLIENT_ID         ?= hospital_dev
FL_DATA_SOURCE       ?= simulated
FL_DB_URL            ?= postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl
PIPELINE_SEQ_LEN     ?= 128
PIPELINE_SAMPLE      ?= 3

# Rede federada real (desktop + notebook)
# FL_SERVER  = IP:porta do desktop (descobrir com: hostname -I)
# FL_HOSPITAL_ID = hospital deste nó (BPSP no desktop, HSL no notebook)
FL_SERVER        ?= localhost:8080
FL_HOSPITAL_ID   ?= BPSP
FL_MIN_CLIENTS   ?= 2
FL_NUM_ROUNDS    ?= 20

# Senha do banco — obrigatória para fl-server e fl-client com dados reais.
# Pode ser definida no .env ou passada diretamente: make fl-server FL_DB_PASSWORD=senha
FL_DB_PASSWORD   ?= senhaForte
FL_DB_PORT       ?= 5432

.PHONY: setup test test-integration test-e2e test-all test-cov experiment clean \
        superlink server-app supernode sim test-pipeline \
        db-up db-down db-wait fl-server fl-client fl-check

setup:
	bash setup.sh

# Unit tests -- no external deps, safe for CI/CD deploy pipeline
test:
	$(PYTEST) tests/unit/ tests/test_fl_cycle_explained.py -v --tb=short

test-cov:
	$(PYTEST) tests/unit/ tests/test_fl_cycle_explained.py -v --tb=short \
		--cov --cov-report=term-missing

# Integration tests -- real components + boundary mocks. Not in deploy pipeline.
test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

# End-to-end test -- real FL cycle, no mocks of our code. Run manually pre-release.
test-e2e:
	$(PYTEST) tests/e2e/ -v --tb=short -m e2e

# Full suite -- unit + integration + e2e
test-all:
	$(PYTEST) tests/ -v --tb=short -m ""

experiment:
	$(PYTHON) experiments/run_experiments_simulation.py

# ── Banco de dados (PostgreSQL via Docker) ────────────────────────────────────

## Sobe o PostgreSQL (TimescaleDB) em background e aguarda estar pronto.
## Necessário antes de fl-server e fl-client quando FL_DB_URL está configurado.
##   make db-up
##   make db-up FL_DB_PASSWORD=outrasenha FL_DB_PORT=5433
db-up:
	FL_DB_PASSWORD="$(FL_DB_PASSWORD)" FL_DB_PORT="$(FL_DB_PORT)" \
	docker compose -f docker-compose.db.yml up -d
	@$(MAKE) --no-print-directory db-wait

## Aguarda o PostgreSQL aceitar conexões (healthcheck do container).
db-wait:
	@echo "Aguardando banco ficar pronto..."
	@until docker compose -f docker-compose.db.yml exec -T db \
	    pg_isready -U mosaicfl -d mosaicfl -q 2>/dev/null; do \
	    printf "."; sleep 2; \
	done
	@echo " banco pronto."

## Para e remove o container do banco (dados preservados no volume).
## Use db-down-v para apagar também os dados.
db-down:
	docker compose -f docker-compose.db.yml down

db-down-v:
	docker compose -f docker-compose.db.yml down -v

# ── Rede federada real (desktop + notebook) ───────────────────────────────────

## Desktop: sobe o banco e inicia o servidor FL.
## Imprime o IP local para configurar o notebook.
##   make fl-server
##   make fl-server FL_NUM_ROUNDS=30 FL_MIN_CLIENTS=2
fl-server: db-up
	FL_DB_URL="$(FL_DB_URL)" \
	FL_HOSPITAL_ID="$(FL_HOSPITAL_ID)" \
	FL_NUM_ROUNDS="$(FL_NUM_ROUNDS)" \
	FL_MIN_CLIENTS="$(FL_MIN_CLIENTS)" \
	$(PYTHON) experiments/run_federated_real.py --mode server --port 8080

## Notebook: sobe o banco local e conecta ao servidor do desktop.
## Substitua FL_SERVER pelo IP impresso pelo fl-server.
##   make fl-client FL_SERVER=192.168.1.100:8080 FL_HOSPITAL_ID=HSL
fl-client: db-up
	FL_DB_URL="$(FL_DB_URL)" \
	FL_HOSPITAL_ID="$(FL_HOSPITAL_ID)" \
	$(PYTHON) experiments/run_federated_real.py --mode client --server "$(FL_SERVER)"

## Verifica conectividade com o servidor (não sobe banco, não inicia cliente).
##   make fl-check FL_SERVER=192.168.1.100:8080
fl-check:
	$(PYTHON) experiments/run_federated_real.py --check --server "$(FL_SERVER)"

# Diagnóstico do SequencePipeline (série temporal de internados HSL+BPSP).
# Sobrescreva variáveis conforme necessário:
#   make test-pipeline FL_DB_URL=postgresql://... PIPELINE_SAMPLE=5
test-pipeline:
	$(PYTHON) scripts/test_pipeline.py \
		--db-url    "$(FL_DB_URL)" \
		--max-seq-len $(PIPELINE_SEQ_LEN) \
		--sample    $(PIPELINE_SAMPLE)

# ── Flower SuperLink (produção) ────────────────────────────────────────────────

# Inicia o SuperLink com TLS. Requer FL_TLS_CERT_DIR com ca.crt/server.crt/server.key.
superlink:
	FL_TLS_CERT_DIR=$(FL_TLS_CERT_DIR) bash scripts/iniciar_servidor_fl.sh

# Dispara ServerApp num SuperLink já em execução (requer flwr run e pyproject.toml).
server-app:
	$(FLWR) run . production

# Inicia um SuperNode (cliente) conectando ao SuperLink.
# Ex: make supernode FL_CLIENT_ID=hospital_1 FL_DATA_SOURCE=sgbd
supernode:
	FL_TLS_CERT_DIR=$(FL_TLS_CERT_DIR) \
	FL_CLIENT_ID=$(FL_CLIENT_ID) \
	FL_DATA_SOURCE=$(FL_DATA_SOURCE) \
	FL_SUPERLINK_ADDRESS=$(FL_SUPERLINK_ADDRESS) \
	bash scripts/iniciar_cliente_fl.sh

# Simulação local com run_simulation (sem SuperLink, sem rede, sem TLS).
sim:
	$(FLWR) run . local-sim

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
