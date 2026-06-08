PYTHON   := .venv/bin/python
PYTEST   := $(PYTHON) -m pytest
FLWR     := .venv/bin/flwr

# Configurações do SuperLink (sobrescrevíveis por variável de ambiente)
FL_TLS_CERT_DIR     ?= certs
FL_SUPERLINK_ADDRESS ?= localhost:9091
FL_CLIENT_ID        ?= hospital_dev
FL_DATA_SOURCE      ?= simulated

.PHONY: setup test test-integration test-e2e test-all test-cov experiment clean \
        superlink server-app supernode sim

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
	$(PYTHON) experiments/run_experiments_v2.py

# ── Flower SuperLink (produção) ────────────────────────────────────────────────

# Inicia o SuperLink com TLS. Requer FL_TLS_CERT_DIR com ca.crt/server.crt/server.key.
superlink:
	FL_TLS_CERT_DIR=$(FL_TLS_CERT_DIR) bash scripts/start_superlink.sh

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
	bash scripts/start_supernode.sh

# Simulação local com run_simulation (sem SuperLink, sem rede, sem TLS).
sim:
	$(FLWR) run . local-sim

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
