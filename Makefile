PYTHON := .venv/bin/python
PYTEST  := $(PYTHON) -m pytest

.PHONY: setup test test-integration test-e2e test-all test-cov run clean

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

run:
	$(PYTHON) experiments/run_experiments_v2.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
