PYTHON := .venv/bin/python
PYTEST  := $(PYTHON) -m pytest

.PHONY: setup test test-cov run clean

setup:
	bash setup.sh

test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ -v --tb=short --cov --cov-report=term-missing

run:
	$(PYTHON) run_experiments_v2.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
