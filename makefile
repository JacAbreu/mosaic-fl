# Makefile
.PHONY: setup run clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .

run:
	.venv/bin/python -m mosaicfl.experiments.runner

clean:
	rm -rf .venv __pycache__ .pytest_cache
