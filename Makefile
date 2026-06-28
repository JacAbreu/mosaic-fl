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

# Simulação cliente — geração e carregamento do seed HSL
# FL_DATA_DIR: diretório com os ZIPs FAPESP (usado no desktop para gerar o seed)
# HSL_SEED:    arquivo .sql.gz gerado no desktop e transferido para o notebook
FL_DATA_DIR      ?= $(HOME)/studies/usp/mba-bigdata-art-int/tcc/data/Dados/Covid-19
HSL_SEED         ?= scripts/db/seeds/hsl_seed.sql.gz
BPSP_SEED        ?= scripts/db/seeds/bpsp_seed.sql.gz

.PHONY: setup test test-integration test-e2e test-all test-cov experiment training training-full clean \
        superlink server-app supernode sim test-pipeline behrt-pooled recalibrate \
        bootstrap-ci seed-sensitivity \
        db-up db-down db-wait fl-server fl-client fl-check \
        client-generate-seed client-db-up client-migrate client-load-hsl client-setup \
        server-generate-seed server-db-reset server-load-bpsp server-setup

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

FL_LOG_TRAINING   ?= experiments/logs/run_complete_$(shell date +%Y%m%d_%H%M%S).log
FL_LOG_SIMULATION ?= experiments/logs/simulation_$(shell date +%Y%m%d_%H%M%S).log

## Treinamento federado com dados reais FAPESP (requer FL_DB_URL)
training:
	FL_ENV=production FL_LOG_FILE="$(FL_LOG_TRAINING)" $(PYTHON) experiments/run_training.py

## Simulação demonstrativa com dados sintéticos (não requer banco)
experiment:
	FL_LOG_FILE="$(FL_LOG_SIMULATION)" $(PYTHON) experiments/run_experiments_simulation.py

## Pooled baseline — BEHRT com pool BPSP+HSL (artefato de pesquisa, nunca em produção)
## Quantifica o custo de privacidade da federação com a mesma arquitetura do modelo FL.
## Requer FL_DB_URL. Saída: experiments/data/behrt_pooled_<timestamp>.json
behrt-pooled:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/run_behrt_pooled.py

## Re-calibração de temperatura sobre o melhor checkpoint salvo no banco.
## Útil quando a calibração falhou (ex: T negativo) sem re-executar o treinamento.
## Requer FL_DB_URL. Saída: experiments/logs/recalibrate_<timestamp>.json
recalibrate:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/run_recalibrate.py

## IC 95% via bootstrap sobre o melhor checkpoint (sem re-treino). Requer FL_DB_URL.
## Persiste em metrics.bootstrap_ci. Saída: experiments/logs/bootstrap_ci_<timestamp>.json
bootstrap-ci:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/run_bootstrap_ci.py

## Sensibilidade a seed: FedAvg vs FedNova, 3 seeds × 30 rounds. Requer FL_DB_URL.
## Checkpoints isolados em SQLite. Resultados em metrics.sensitivity_runs.
## Saída: experiments/logs/seed_sensitivity_<timestamp>.json
## Opções: make seed-sensitivity ROUNDS=20 SEEDS="42 7"
ROUNDS ?= 30
SEEDS  ?= 42 7 123
seed-sensitivity:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/run_seed_sensitivity.py \
		--rounds $(ROUNDS) --seeds $(SEEDS)

## Treinamento completo + pooled baseline em sequência, em arquivo de log único.
## FL_LOG_FILE é definido uma vez e passado para ambos os scripts via variável de shell.
training-full:
	@LOG="experiments/logs/run_complete_$$(date +%Y%m%d_%H%M%S).log"; \
	FL_ENV=production FL_LOG_FILE="$$LOG" $(PYTHON) experiments/run_training.py; \
	FL_DB_URL="$(FL_DB_URL)" FL_LOG_FILE="$$LOG" $(PYTHON) experiments/run_behrt_pooled.py

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

# ── Simulação cliente — seed HSL ──────────────────────────────────────────────
#
# DESKTOP (onde estão os ZIPs FAPESP):
#   make client-generate-seed                  # gera scripts/db/seeds/hsl_seed.sql.gz
#   make client-generate-seed FL_DATA_DIR=...  # se os ZIPs estiverem em outro diretório
#   # Transfira hsl_seed.sql.gz para o notebook (git, scp, pendrive etc.)
#
# NOTEBOOK (cliente FL — após receber o arquivo):
#   make client-setup                          # sobe banco + aplica migrations + carrega HSL
#

## Gera o seed SQL do HSL a partir dos ZIPs FAPESP (executar no DESKTOP).
## Produz scripts/db/seeds/hsl_seed.sql.gz (~20-40 MB comprimido).
##   make client-generate-seed
##   make client-generate-seed FL_DATA_DIR=/outro/caminho FL_DB_URL=postgresql://...
client-generate-seed:
	$(PYTHON) scripts/db/generate_hsl_seed.py \
		--data-dir "$(FL_DATA_DIR)" \
		--output   "$(HSL_SEED)" \
		$(if $(FL_DB_URL),--db-url "$(FL_DB_URL)",)

## Sobe o PostgreSQL do cliente (mesmo compose do servidor — banco isolado).
client-db-up:
	FL_DB_PASSWORD="$(FL_DB_PASSWORD)" FL_DB_PORT="$(FL_DB_PORT)" \
	docker compose -f docker-compose.db.yml up -d
	@$(MAKE) --no-print-directory db-wait

## Aplica todas as migrations (001→010) no banco do cliente.
## A migration 010 registra este nó como cliente HSL da simulação.
client-migrate:
	FL_DB_URL="$(FL_DB_URL)" bash scripts/db/migrate.sh upgrade head

## Carrega o seed HSL no banco do cliente.
## Requer que hsl_seed.sql.gz exista em $(HSL_SEED).
client-load-hsl:
	@test -f "$(HSL_SEED)" || \
	  (echo "ERRO: $(HSL_SEED) não encontrado." \
	       "Gere no desktop com 'make client-generate-seed' e transfira para este equipamento." \
	   && exit 1)
	@echo "Carregando $(HSL_SEED) no banco..."
	zcat "$(HSL_SEED)" | \
	  docker exec -i mosaicfl-db \
	    psql -U mosaicfl -d mosaicfl -v ON_ERROR_STOP=1
	@echo "Seed HSL carregado com sucesso."

## Sequência completa para o notebook cliente:
##   1. Sobe o banco Docker
##   2. Aplica migrations 001→010
##   3. Carrega dados do HSL
##   make client-setup
##   make client-setup FL_DB_PASSWORD=outrasenha HSL_SEED=outro/caminho.sql.gz
client-setup: client-db-up client-migrate client-load-hsl
	@echo "Cliente HSL pronto. Execute 'make fl-client FL_SERVER=<IP_DESKTOP>:8080' para iniciar o treinamento federado."

# ── Simulação servidor — seed BPSP ────────────────────────────────────────────
#
# Por que BPSP e não Einstein?
#   O Einstein possui mais exames (3,4M), mas não tem arquivo de desfechos.
#   O SequencePipeline exige desfechos para gerar os labels de prognóstico.
#   O BPSP tem o maior volume dentre os hospitais com desfechos (6,3M exames).
#
# DESKTOP (servidor FL):
#   make server-generate-seed   # gera scripts/db/seeds/bpsp_seed.sql.gz
#   make server-setup           # apaga dados anteriores + aplica migrations + carrega BPSP
#

## Gera o seed SQL do BPSP a partir dos ZIPs FAPESP (executar no DESKTOP).
## Produz scripts/db/seeds/bpsp_seed.sql.gz.
##   make server-generate-seed
##   make server-generate-seed FL_DB_URL=postgresql://... FL_DATA_DIR=/outro/caminho
server-generate-seed:
	$(PYTHON) scripts/db/generate_bpsp_seed.py \
		--data-dir "$(FL_DATA_DIR)" \
		--output   "$(BPSP_SEED)" \
		$(if $(FL_DB_URL),--db-url "$(FL_DB_URL)",)

## Apaga todos os dados clínicos do banco do servidor (preserva schema e migrations).
## Use antes de recarregar com um novo hospital.
server-db-reset:
	@echo "Apagando dados clínicos do banco do servidor..."
	docker exec -i mosaicfl-db psql -U mosaicfl -d mosaicfl -v ON_ERROR_STOP=1 <<'SQL'
	TRUNCATE clinical.patients CASCADE;
	TRUNCATE metrics.risk_history;
	SQL
	@echo "Banco resetado. Schema e migrations preservados."

## Carrega o seed BPSP no banco do servidor.
server-load-bpsp:
	@test -f "$(BPSP_SEED)" || \
	  (echo "ERRO: $(BPSP_SEED) não encontrado. Execute 'make server-generate-seed' primeiro." \
	   && exit 1)
	@echo "Carregando $(BPSP_SEED) no banco..."
	zcat "$(BPSP_SEED)" | \
	  docker exec -i mosaicfl-db \
	    psql -U mosaicfl -d mosaicfl -v ON_ERROR_STOP=1
	@echo "Seed BPSP carregado com sucesso."

## Sequência completa para o servidor desktop:
##   1. Sobe o banco Docker (se não estiver rodando)
##   2. Aplica migrations 001→010
##   3. Apaga dados anteriores
##   4. Carrega dados do BPSP
##   make server-setup
##   make server-setup FL_DB_PASSWORD=outrasenha
server-setup: db-up client-migrate server-db-reset server-load-bpsp
	@echo "Servidor BPSP pronto. Execute 'make fl-server' para iniciar o treinamento federado."

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
