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
FL_API_HOST      ?= 0.0.0.0
FL_API_PORT      ?= 8000

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

# Nome do container Postgres alvo dos alvos server-db-reset/server-load-bpsp/
# client-db-reset/client-load-hsl. Default = container do docker-compose.db.yml
# padrão. Se você criou um container separado (nome customizado, ex.:
# mosaicfl-db-bpsp — ver docs/Tutorial_Rede_Federada_Real_Desktop_Notebook.md),
# sobrescreva explicitamente: make server-db-reset FL_DB_CONTAINER=mosaicfl-db-bpsp
# — sem isso, esses alvos sempre operam no container padrão, mesmo que você
# pretenda atingir outro (risco real: truncar o banco errado).
FL_DB_CONTAINER  ?= mosaicfl-db

# Nome do banco de dados Postgres (não do container) usado por
# server-load-bpsp/client-load-hsl/server-db-reset/client-db-reset — sobrescreva
# junto com FL_DB_CONTAINER quando o container de destino tiver um banco com
# nome diferente de "mosaicfl" (ex.: full-db-* usa FL_DB_NAME=$(FULL_DB_NAME)).
FL_DB_NAME       ?= mosaicfl

# Container Postgres para o banco "completo" (todos os dados FAPESP — BPSP +
# HSL — combinados, isolado do mosaicfl-db principal). Nome do banco em si é
# diferente de "mosaicfl" de propósito, para não confundir com o principal.
# Ver alvos full-db-*.
FULL_DB_CONTAINER ?= mosaic-brute-data
FULL_DB_NAME      ?= mosaic_brute_data
FULL_DB_PORT      ?= 5434

.PHONY: setup ollama-setup ollama-check \
        test test-integration test-e2e test-all test-cov experiment training training-full \
        training-full-cuda training-bpsp-only training-hsl-only \
        training-iid-contrast training-iid-contrast-cuda \
        training-dp-curve training-dp-curve-cuda training-dp-curve-replicas-cuda \
        training-dp-curve-deterministic-cuda clean \
        superlink server-app supernode sim test-pipeline behrt-pooled recalibrate \
        bootstrap-ci seed-sensitivity \
        db-up db-down db-wait fl-server fl-client fl-check \
        client-generate-seed client-db-up client-migrate client-load-hsl client-setup client-db-reset \
        server-generate-seed server-db-reset server-load-bpsp server-setup \
        api export-checkpoint

setup:
	bash setup.sh

## Instala o Ollama e faz o pull do modelo LLM configurado (FL_LLM_MODEL).
## Requer conexão com a internet. Idempotente: seguro de reexecutar.
##   make ollama-setup
##   make ollama-setup FL_LLM_MODEL=llama3.2:3b
ollama-setup:
	@echo "=== Verificando instalação do Ollama ==="
	@if command -v ollama > /dev/null 2>&1; then \
		echo "  Ollama já instalado: $$(ollama --version)"; \
	else \
		echo "  Instalando Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	fi
	@echo "=== Iniciando ollama serve (background) ==="
	@pgrep -x ollama > /dev/null 2>&1 || (ollama serve > /tmp/ollama.log 2>&1 &); \
	sleep 3
	@echo "=== Baixando modelo $(FL_LLM_MODEL) ==="
	ollama pull $(FL_LLM_MODEL)
	@echo "=== Ollama pronto: $(FL_LLM_MODEL) disponível ==="

## Verifica se o Ollama está online e o modelo está disponível.
## Retorna exit code 0 se OK, 1 se não disponível (útil em CI).
ollama-check:
	@echo "=== Verificando Ollama ==="
	@curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && \
		echo "  Ollama: online" || \
		(echo "  Ollama: OFFLINE — execute 'ollama serve' ou 'make ollama-setup'"; exit 1)
	@ollama list 2>/dev/null | grep -q "$(FL_LLM_MODEL)" && \
		echo "  Modelo $(FL_LLM_MODEL): disponível" || \
		(echo "  Modelo $(FL_LLM_MODEL): NÃO encontrado — execute 'ollama pull $(FL_LLM_MODEL)'"; exit 1)
	@echo "=== OK ==="

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
	FL_ENV=production FL_LOG_FILE="$(FL_LOG_TRAINING)" $(PYTHON) experiments/training_runner/run_training.py

## Simulação demonstrativa com dados sintéticos (não requer banco)
experiment:
	FL_LOG_FILE="$(FL_LOG_SIMULATION)" $(PYTHON) experiments/training_runner/run_experiments_simulation.py

## Pooled baseline — BEHRT com pool BPSP+HSL (artefato de pesquisa, nunca em produção)
## Quantifica o custo de privacidade da federação com a mesma arquitetura do modelo FL.
## Requer FL_DB_URL. Saída: experiments/data/behrt_pooled_<timestamp>.json
behrt-pooled:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/training_runner/run_behrt_pooled.py

## Re-calibração de temperatura sobre o melhor checkpoint salvo no banco.
## Útil quando a calibração falhou (ex: T negativo) sem re-executar o treinamento.
## Requer FL_DB_URL. Saída: experiments/logs/recalibrate_<timestamp>.json
recalibrate:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/training_runner/run_recalibrate.py

## IC 95% via bootstrap sobre o melhor checkpoint (sem re-treino). Requer FL_DB_URL.
## Persiste em metrics.bootstrap_ci. Saída: experiments/logs/bootstrap_ci_<timestamp>.json
bootstrap-ci:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/training_runner/run_bootstrap_ci.py

## Sensibilidade a seed: FedAvg vs FedNova, 3 seeds × 30 rounds. Requer FL_DB_URL.
## Checkpoints isolados em SQLite. Resultados em metrics.sensitivity_runs.
## Saída: experiments/logs/seed_sensitivity_<timestamp>.json
## Opções: make seed-sensitivity ROUNDS=20 SEEDS="42 7"
ROUNDS ?= 30
SEEDS  ?= 42 7 123
seed-sensitivity:
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) experiments/training_runner/run_seed_sensitivity.py \
		--rounds $(ROUNDS) --seeds $(SEEDS)

## Classifica o(s) treinamento(s) no banco: "ajuste" (default — NÃO citar como
## resultado final) ou "treinamento_real" (resultado formal para o TCC).
## Uso: FL_RUN_CLASSIFICATION=treinamento_real make training-full-cuda
FL_RUN_CLASSIFICATION ?= ajuste

## Treinamento federado com apenas clientes BPSP (leave-one-client-out).
## Test/cal sempre no set global (BPSP+HSL) para comparação justa. Requer FL_DB_URL.
training-bpsp-only:
	@LOG="experiments/logs/training_bpsp_only_$$(date +%Y%m%d_%H%M%S).log"; \
	FL_ENV=production FL_INCLUDE_HOSPITALS=BPSP FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" \
	$(PYTHON) experiments/training_runner/run_training.py

## Treinamento federado com apenas clientes HSL (leave-one-client-out).
## Test/cal sempre no set global (BPSP+HSL) para comparação justa. Requer FL_DB_URL.
training-hsl-only:
	@LOG="experiments/logs/training_hsl_only_$$(date +%Y%m%d_%H%M%S).log"; \
	FL_ENV=production FL_INCLUDE_HOSPITALS=HSL FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" \
	$(PYTHON) experiments/training_runner/run_training.py

## Treinamento federado com partição IID simulada (pool BPSP+HSL embaralhado,
## clientes virtuais) — Experimento 3: contraste causal non-IID real (fase 3)
## vs. IID simulado, mesmo algoritmo/hiperparâmetros/seed. Requer FL_DB_URL.
training-iid-contrast:
	@LOG="experiments/logs/training_iid_contrast_$$(date +%Y%m%d_%H%M%S).log"; \
	FL_ENV=production FL_PARTITION_MODE=iid_simulado FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" \
	$(PYTHON) experiments/training_runner/run_training.py

## Mesma coisa, forçando GPU.
training-iid-contrast-cuda:
	@LOG="experiments/logs/training_iid_contrast_cuda_$$(date +%Y%m%d_%H%M%S).log"; \
	FL_ENV=production FL_DEVICE=cuda FL_PARTITION_MODE=iid_simulado FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" \
	$(PYTHON) experiments/training_runner/run_training.py

## Pipeline completo: BPSP-only → HSL-only → federado (BPSP+HSL) → pooled baseline → IID simulado (contraste).
## Sem parametrização externa — env vars definidas internamente.
## Requer FL_DB_URL, dados de ambos os hospitais no banco e Ollama rodando com gemma3:4b.
## Para trocar o modelo LLM: FL_LLM_MODEL=outro-modelo make training-full
## Para marcar como resultado oficial: FL_RUN_CLASSIFICATION=treinamento_real make training-full
FL_LLM_BACKEND ?= ollama
FL_LLM_MODEL   ?= gemma3:4b
FL_DP_NOISE    ?= 0.0   # σ do ruído DP (0.0 = DP desabilitado)
FL_DP_CLIP     ?= 1.0   # S = norma máxima do update do cliente

training-full:
	@LOG="experiments/logs/run_complete_$$(date +%Y%m%d_%H%M%S).log"; \
	echo "=== 1/5 training-bpsp-only ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_INCLUDE_HOSPITALS=BPSP FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 2/5 training-hsl-only ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_INCLUDE_HOSPITALS=HSL FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 3/5 training-federated (BPSP+HSL, non-IID real) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 4/5 behrt-pooled baseline ===" | tee -a "$$LOG"; \
	FL_DB_URL="$(FL_DB_URL)" FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_behrt_pooled.py; \
	echo "=== 5/5 training-federated (IID simulado — contraste Experimento 3) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_PARTITION_MODE=iid_simulado FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py

## Pipeline completo (mesmas 5 fases de training-full), forçando execução na GPU via FL_DEVICE=cuda.
## Requer FL_DB_URL, dados de ambos os hospitais no banco, Ollama rodando com gemma3:4b e CUDA disponível
## (validar antes com: .venv/bin/python -c "import torch; print(torch.cuda.is_available())").
## Para marcar como resultado oficial: FL_RUN_CLASSIFICATION=treinamento_real make training-full-cuda
training-full-cuda:
	@LOG="experiments/logs/run_complete_cuda_$$(date +%Y%m%d_%H%M%S).log"; \
	echo "=== 1/5 training-bpsp-only (cuda) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_DEVICE=cuda FL_INCLUDE_HOSPITALS=BPSP FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 2/5 training-hsl-only (cuda) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_DEVICE=cuda FL_INCLUDE_HOSPITALS=HSL FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 3/5 training-federated (BPSP+HSL, non-IID real, cuda) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_DEVICE=cuda FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	echo "=== 4/5 behrt-pooled baseline (cuda) ===" | tee -a "$$LOG"; \
	FL_DEVICE=cuda FL_DB_URL="$(FL_DB_URL)" FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_behrt_pooled.py; \
	echo "=== 5/5 training-federated (IID simulado — contraste Experimento 3, cuda) ===" | tee -a "$$LOG"; \
	FL_ENV=production FL_DEVICE=cuda FL_PARTITION_MODE=iid_simulado FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_DP_NOISE=$(FL_DP_NOISE) FL_DP_CLIP=$(FL_DP_CLIP) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py

## Curva Acurácia × ε (DP-FedAvg, Exp 17/18/19): roda só a fase Federada
## (BPSP+HSL, non-IID real) — onde a privacidade client-level do DP-FedAvg de
## fato se aplica. BPSP-only/HSL-only são cliente único (sem agregação entre
## clientes) e o Pooled/RF não passam por FL — não fazem parte desta curva.
## 3 execuções, σ=1,0/0,5/2,0, S=1,0 fixo (mesmos valores do plano original em
## docs/TODO.md). Cada σ gera log e training_id próprios, todos marcados com
## partition_mode=natural e dp_noise_multiplier preenchido.
## Consultar depois com:
##   SELECT id, dp_noise_multiplier, dp_epsilon_simple, dp_epsilon_rdp,
##          best_accuracy, macro_f1 FROM metrics.fl_trainings
##   WHERE dp_noise_multiplier IS NOT NULL ORDER BY id;
training-dp-curve:
	@for SIGMA in 1.0 0.5 2.0; do \
		LOG="experiments/logs/dp_curve_sigma$${SIGMA}_$$(date +%Y%m%d_%H%M%S).log"; \
		echo "=== DP curve sigma=$$SIGMA (S=1.0) ===" | tee -a "$$LOG"; \
		FL_ENV=production FL_DP_NOISE=$$SIGMA FL_DP_CLIP=1.0 FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	done

## Mesma coisa, forçando GPU — recomendado: qualidade já confirmada equivalente
## entre devices (ver docs/Sumario_Treinamento_Parte3.md), e ~10x mais rápido
## por execução — relevante aqui porque são 3 execuções, não 1.
training-dp-curve-cuda:
	@for SIGMA in 1.0 0.5 2.0; do \
		LOG="experiments/logs/dp_curve_sigma$${SIGMA}_cuda_$$(date +%Y%m%d_%H%M%S).log"; \
		echo "=== DP curve sigma=$$SIGMA (S=1.0, cuda) ===" | tee -a "$$LOG"; \
		FL_ENV=production FL_DEVICE=cuda FL_DP_NOISE=$$SIGMA FL_DP_CLIP=1.0 FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	done

## Investigação (2026-07-03): a curva original (training-dp-curve-cuda, ids
## 48-50) teve relação ruído×utilidade NÃO monotônica (σ=1,0 pior que σ=2,0,
## ver docs/Sumario_Treinamento_Parte3.md). Hipótese: variância de execução
## única, não efeito real do ruído. Roda cada σ com 3 seeds (42 — a mesma da
## curva original — mais 43 e 44) para ver se o padrão se mantém ou é
## específico da seed=42. Data continua idêntica entre réplicas (dataloaders.py
## usa geradores fixos em RANDOM_SEED, não afetados por FL_RANDOM_SEED) — só
## variam inicialização de pesos e sorteios de ruído do DP.
## Default FL_RUN_CLASSIFICATION=ajuste (investigação de robustez, não resultado
## novo) — sobrescrever explicitamente se as réplicas forem citadas como dado formal.
training-dp-curve-replicas-cuda:
	@for SIGMA in 1.0 0.5 2.0; do \
		for SEED in 42 43 44; do \
			LOG="experiments/logs/dp_replica_sigma$${SIGMA}_seed$${SEED}_$$(date +%Y%m%d_%H%M%S).log"; \
			echo "=== DP replica sigma=$$SIGMA seed=$$SEED (cuda) ===" | tee -a "$$LOG"; \
			FL_ENV=production FL_DEVICE=cuda FL_DP_NOISE=$$SIGMA FL_DP_CLIP=1.0 FL_RANDOM_SEED=$$SEED FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
		done; \
	done

## Etapa 2 da investigação de não-monotonicidade (após Etapa 1 — réplicas com
## seeds diferentes — confirmar que σ=1,0 é consistentemente o pior ponto).
## Reexecuta os mesmos 3 σ, MESMA seed=42 da curva original (ids 48-50), com
## torch.use_deterministic_algorithms(True, warn_only=True) ativo. Objetivo:
## saber se a curva original já era reprodutível bit-a-bit (resultado bate com
## ids 48-50) ou se ainda há não-determinismo de GPU não coberto por
## cudnn.deterministic=True sozinho (resultado diverge).
training-dp-curve-deterministic-cuda:
	@for SIGMA in 1.0 0.5 2.0; do \
		LOG="experiments/logs/dp_deterministic_sigma$${SIGMA}_$$(date +%Y%m%d_%H%M%S).log"; \
		echo "=== DP deterministic sigma=$$SIGMA seed=42 (cuda) ===" | tee -a "$$LOG"; \
		FL_ENV=production FL_DEVICE=cuda FL_DP_NOISE=$$SIGMA FL_DP_CLIP=1.0 FL_DETERMINISTIC=1 FL_LLM_BACKEND=$(FL_LLM_BACKEND) FL_LLM_MODEL=$(FL_LLM_MODEL) FL_RUN_CLASSIFICATION=$(FL_RUN_CLASSIFICATION) FL_LOG_FILE="$$LOG" $(PYTHON) experiments/training_runner/run_training.py; \
	done

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
	$(PYTHON) experiments/training_runner/run_federated_real.py --mode server --port 8080

## Notebook: sobe o banco local e conecta ao servidor do desktop.
## Substitua FL_SERVER pelo IP impresso pelo fl-server.
##   make fl-client FL_SERVER=192.168.1.100:8080 FL_HOSPITAL_ID=HSL
fl-client: db-up
	FL_DB_URL="$(FL_DB_URL)" \
	FL_HOSPITAL_ID="$(FL_HOSPITAL_ID)" \
	$(PYTHON) experiments/training_runner/run_federated_real.py --mode client --server "$(FL_SERVER)"

## Verifica conectividade com o servidor (não sobe banco, não inicia cliente).
##   make fl-check FL_SERVER=192.168.1.100:8080
fl-check:
	$(PYTHON) experiments/training_runner/run_federated_real.py --check --server "$(FL_SERVER)"

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
# root-certificates é lido de ~/.flwr/config.toml (migrado automaticamente pelo
# flwr na primeira execução), NÃO de FL_TLS_CERT_DIR — tentativa de passar
# root-certificates via --federation-config falha nesta versão do flwr (1.30.0)
# com "Unknown simulation config field(s): root_certificates", aparentemente por
# essa flag esperar o schema de simulação, não o de SuperLink (revertido em
# 2026-07-05 após causar regressão real — ver docs/Linha_do_Tempo_MOSAIC-FL.md).
# Se você mover FL_TLS_CERT_DIR de lugar, atualize manualmente o caminho em
# ~/.flwr/config.toml (seção [superlink.production], chave root-certificates).
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
	@echo "Carregando $(HSL_SEED) no banco ($(FL_DB_CONTAINER))..."
	zcat "$(HSL_SEED)" | \
	  docker exec -i $(FL_DB_CONTAINER) \
	    psql -U mosaicfl -d $(FL_DB_NAME) -v ON_ERROR_STOP=1
	@echo "Seed HSL carregado com sucesso."
	@echo "Calculando referências canônicas e preenchendo classification (backfill)..."
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) scripts/compute_analyte_references.py
	@echo "Backfill concluído — banco pronto para o treinamento federado."

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
## Use antes de recarregar com um novo hospital, ou para recarregar o mesmo seed
## após regenerá-lo (ex.: depois de uma correção no gerador) — sem isso, a carga
## falha com "duplicate key value violates unique constraint" (patients_pkey).
server-db-reset:
	@echo "Apagando dados clínicos do banco do servidor ($(FL_DB_CONTAINER))..."
	docker exec -i $(FL_DB_CONTAINER) psql -U mosaicfl -d $(FL_DB_NAME) -v ON_ERROR_STOP=1 -c "TRUNCATE clinical.patients CASCADE; TRUNCATE metrics.clinical_outcomes; TRUNCATE metrics.exam_records; TRUNCATE metrics.risk_history;"
	@echo "Banco resetado. Schema e migrations preservados."

## Apaga todos os dados clínicos do banco do cliente (preserva schema e migrations).
## Equivalente ao server-db-reset, para o notebook. Mesma justificativa: use antes
## de recarregar um seed regenerado.
client-db-reset:
	@echo "Apagando dados clínicos do banco do cliente ($(FL_DB_CONTAINER))..."
	docker exec -i $(FL_DB_CONTAINER) psql -U mosaicfl -d $(FL_DB_NAME) -v ON_ERROR_STOP=1 -c "TRUNCATE clinical.patients CASCADE; TRUNCATE metrics.clinical_outcomes; TRUNCATE metrics.exam_records; TRUNCATE metrics.risk_history;"
	@echo "Banco resetado. Schema e migrations preservados."

## Carrega o seed BPSP no banco do servidor.
server-load-bpsp:
	@test -f "$(BPSP_SEED)" || \
	  (echo "ERRO: $(BPSP_SEED) não encontrado. Execute 'make server-generate-seed' primeiro." \
	   && exit 1)
	@echo "Carregando $(BPSP_SEED) no banco ($(FL_DB_CONTAINER))..."
	zcat "$(BPSP_SEED)" | \
	  docker exec -i $(FL_DB_CONTAINER) \
	    psql -U mosaicfl -d $(FL_DB_NAME) -v ON_ERROR_STOP=1
	@echo "Seed BPSP carregado com sucesso."
	@echo "Calculando referências canônicas e preenchendo classification (backfill)..."
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) scripts/compute_analyte_references.py
	@echo "Backfill concluído."

## Sequência completa para o servidor desktop:
##   1. Sobe o banco Docker (se não estiver rodando)
##   2. Aplica migrations 001→010
##   3. Apaga dados anteriores
##   4. Carrega dados do BPSP
##   make server-setup
##   make server-setup FL_DB_PASSWORD=outrasenha
server-setup: db-up client-migrate server-db-reset server-load-bpsp
	@echo "Servidor BPSP pronto. Execute 'make fl-server' para iniciar o treinamento federado."

# ── Banco completo — todos os dados FAPESP (BPSP + HSL) num container isolado ─
#
# Container Docker próprio (docker run direto, fora do docker-compose.db.yml
# padrão — mesmo padrão do mosaicfl-db-bpsp do tutorial de rede real). O banco
# Postgres interno se chama "mosaic_brute_data" (FULL_DB_NAME) — de propósito
# diferente de "mosaicfl", para não confundir os dois bancos.
#
# Uso (sequência completa):
#   make full-db-setup
#   make full-db-setup FULL_DB_CONTAINER=outro_nome FULL_DB_NAME=outro_db FULL_DB_PORT=5435
#
# Ou passo a passo (para reexecutar só uma etapa):
#   make full-db-up
#   make full-db-generate-seeds
#   make full-db-migrate FL_DB_URL=postgresql://mosaicfl:senhaForte@localhost:5434/mosaic_brute_data
#   make full-db-load    FL_DB_URL=postgresql://mosaicfl:senhaForte@localhost:5434/mosaic_brute_data

## Sobe um Postgres novo e isolado (container customizado, fora do compose padrão).
full-db-up:
	docker run -d \
		--name $(FULL_DB_CONTAINER) \
		-e POSTGRES_DB=$(FULL_DB_NAME) -e POSTGRES_USER=mosaicfl \
		-e POSTGRES_PASSWORD=$(FL_DB_PASSWORD) \
		-p $(FULL_DB_PORT):5432 \
		-v $(FULL_DB_CONTAINER)_data:/home/postgres/pgdata/data \
		timescale/timescaledb-ha:pg16
	@echo "Aguardando banco '$(FULL_DB_CONTAINER)' ficar pronto..."
	@until docker exec $(FULL_DB_CONTAINER) pg_isready -U mosaicfl -d $(FULL_DB_NAME) >/dev/null 2>&1; do sleep 1; done
	@echo "Banco '$(FULL_DB_CONTAINER)' pronto na porta $(FULL_DB_PORT), banco '$(FULL_DB_NAME)'."

## Gera os dois seeds (BPSP e HSL) a partir dos ZIPs FAPESP em $(FL_DATA_DIR).
## Sobrescreve scripts/db/seeds/{bpsp,hsl}_seed.sql.gz caso já existam — garante
## que o banco completo carregue a versão mais atual dos geradores, não seeds
## antigos possivelmente gerados por uma versão anterior do código.
full-db-generate-seeds:
	$(MAKE) --no-print-directory server-generate-seed
	$(MAKE) --no-print-directory client-generate-seed

## Aplica as migrations no banco completo (requer FL_DB_URL apontando pra ele).
full-db-migrate:
	FL_DB_URL="$(FL_DB_URL)" bash scripts/db/migrate.sh upgrade head

## Carrega BPSP + HSL (seeds já gerados) no banco completo — server-load-bpsp/
## client-load-hsl já incluem o backfill de classification automaticamente.
## Ao final, gera o vocabulário padrão compartilhado (checkpoints/standard_vocab.json),
## necessário para treinamento federado real via Caminho B (ver
## docs/Tutorial_Rede_Federada_Real_Desktop_Notebook.md, seção 1.4b).
full-db-load:
	$(MAKE) --no-print-directory server-load-bpsp FL_DB_CONTAINER=$(FULL_DB_CONTAINER) FL_DB_NAME=$(FULL_DB_NAME) FL_DB_URL="$(FL_DB_URL)"
	$(MAKE) --no-print-directory client-load-hsl  FL_DB_CONTAINER=$(FULL_DB_CONTAINER) FL_DB_NAME=$(FULL_DB_NAME) FL_DB_URL="$(FL_DB_URL)"
	mkdir -p checkpoints
	FL_DB_URL="$(FL_DB_URL)" $(PYTHON) scripts/build_standard_vocab.py --output checkpoints/standard_vocab.json

## Sequência completa: sobe o container, gera os seeds, aplica migrations,
## carrega BPSP+HSL e gera o vocabulário padrão compartilhado.
full-db-setup:
	$(MAKE) --no-print-directory full-db-up
	$(MAKE) --no-print-directory full-db-generate-seeds
	$(MAKE) --no-print-directory full-db-migrate FL_DB_URL="postgresql://mosaicfl:$(FL_DB_PASSWORD)@localhost:$(FULL_DB_PORT)/$(FULL_DB_NAME)"
	$(MAKE) --no-print-directory full-db-load    FL_DB_URL="postgresql://mosaicfl:$(FL_DB_PASSWORD)@localhost:$(FULL_DB_PORT)/$(FULL_DB_NAME)"
	@echo ""
	@echo "Banco completo pronto: postgresql://mosaicfl:$(FL_DB_PASSWORD)@localhost:$(FULL_DB_PORT)/$(FULL_DB_NAME)"

## Inicia a API de inferência REST.
## O modelo é carregado automaticamente do banco (FL_DB_URL) se não houver checkpoint em arquivo.
##   make api
##   make api FL_API_PORT=9000 FL_AUTH_REQUIRED=false
##   make api FL_AUTH_REQUIRED=false FL_ENV=development
api: db-up
	$(PYTHON) -m infrastructure.mosaicfl_api \
	    --host $(FL_API_HOST) \
	    --port $(FL_API_PORT)

## Exporta o melhor checkpoint do banco (PostgreSQL/SQLite) para checkpoints/best_model.pt
##
## QUANDO USAR: a API carrega o modelo diretamente do banco via CheckpointStore na maioria
## dos casos (make api já faz isso automaticamente). Use export-checkpoint apenas quando:
##   - Quiser implantar a API em outro servidor sem acesso ao banco de treinamento
##   - Quiser inspecionar ou arquivar os pesos manualmente (ex: antes da defesa)
##   - A API estiver configurada com FL_CHECKPOINT_DIR apontando para um diretório local
##
## O arquivo gerado NÃO é versionado no git (.gitignore). O banco é a fonte da verdade.
## Para recriar o checkpoint a qualquer momento: make training-full && make export-checkpoint
##
##   make export-checkpoint
##   make export-checkpoint FL_TRAINING_ID=5   # exporta treinamento específico
export-checkpoint:
	@mkdir -p checkpoints
	$(PYTHON) scripts/export_checkpoint.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
