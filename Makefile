.PHONY: help test reproduce local-validate 4090-benchmark slides

PYTHON := .venv/Scripts/python.exe

help:
	@echo "ACC LLM Enhancement — Make targets"
	@echo "  make test            Run unit tests"
	@echo "  make local-validate  Run small local benchmark on 1.5B"
	@echo "  make 4090-benchmark  Run publication-scale 4090 benchmark (requires 4090 + API key)"
	@echo "  make slides          Build weekly update slides"
	@echo "  make reproduce       Run full reproduction pipeline"

test:
	$(PYTHON) -m pytest tests/ -q

local-validate:
	bash scripts/run_local_validation.sh

4090-benchmark:
	bash scripts/run_4090_benchmark.sh Qwen/Qwen2.5-7B openai gpt-4o-mini

slides:
	$(PYTHON) scripts/build_weekly_slides.py

reproduce:
	bash scripts/reproduce.sh Qwen/Qwen2.5-7B
