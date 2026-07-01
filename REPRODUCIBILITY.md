# Reproducibility Guide

This document provides exact commands, seeds, and environment details to reproduce the ACC hallucination-detection experiments.

## Environment

### Hardware Used for Development

- **Local development:** Intel Arc 140T (16 GB unified memory), Windows 11, Python 3.11.6
- **Target large-scale eval:** NVIDIA RTX 4090 (24 GB VRAM), Linux, Python 3.10+

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For CUDA / RTX 4090, install PyTorch with CUDA first:

```bash
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For Intel XPU (local):

```bash
pip install -r requirements-xpu.txt
```

### Dependency Versions (Validated)

See `requirements-pinned.txt` for an exact snapshot of a working environment.

To generate a fresh pinned file:

```bash
pip freeze > requirements-pinned.txt
```

## Data

### HaluEval

Downloaded automatically by `scripts/download_pubmedqa.py` or place manually at:

```
data/halueval/data.jsonl
```

### TruthfulQA and PubMedQA

Downloaded on first run via the `datasets` library from HuggingFace.

## Models

### Qwen2.5-1.5B (local development)

```bash
python scripts/download_model_robust_hf.py Qwen/Qwen2.5-1.5B --local-dir models/qwen2.5-1.5b
```

### Qwen2.5-7B (RTX 4090 target)

```bash
python scripts/download_model_robust_hf.py Qwen/Qwen2.5-7B --local-dir models/Qwen_Qwen2.5-7B
```

## Reproduction Commands

### 1. Unit Tests

```bash
pytest tests/ -q
```

Expected: 99 passed.

### 2. Small-Scale Benchmark (1.5B, no judge)

```bash
python scripts/run_benchmark_eval.py \
    --model models/qwen2.5-1.5b \
    --halueval 10 \
    --truthfulqa 10 \
    --pubmedqa 10 \
    --max-new-tokens 15 \
    --seed 42 \
    --output results/benchmark_eval_1.5b_repro.json
```

### 3. Ablation Study (1.5B)

```bash
python scripts/run_ablation.py \
    --model models/qwen2.5-1.5b \
    --max-new-tokens 15 \
    --seed 42 \
    --output results/ablation_study_1.5b_repro.json
```

### 4. Publication-Scale Benchmark (7B, RTX 4090)

On the 4090 machine:

```bash
bash scripts/run_4090_benchmark.sh Qwen/Qwen2.5-7B
```

This is equivalent to:

```bash
python scripts/setup_4090.py --model Qwen/Qwen2.5-7B
python scripts/run_benchmark_eval.py \
    --model models/Qwen_Qwen2.5-7B \
    --halueval 200 \
    --truthfulqa 200 \
    --pubmedqa 100 \
    --max-new-tokens 30 \
    --seed 42 \
    --use-llm-judge \
    --judge-type openai \
    --openai-model gpt-4o-mini \
    --output results/benchmark_eval_Qwen_Qwen2.5-7B_500samples.json
```

**Note:** For the OpenAI judge, set `OPENAI_API_KEY` in the environment.

For Anthropic judge:

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_benchmark_eval.py \
    ... \
    --judge-type anthropic \
    --anthropic-model claude-3-5-sonnet-20241022
```

## Random Seeds

- Dataset sampling seeds:
  - HaluEval: `--seed 42`
  - TruthfulQA: `--seed 43`
  - PubMedQA: `--seed 44`
- Generation seeds: `--seed 42 + sample_index`
- SAPLMA train/val split: `--seed 42`
- Bootstrap CI: fixed RNG seed 42

## Known Limitations

- Local LLM-as-judge loads a second copy of the model and is slow on XPU. Use API judges for publication-scale runs.
- `SelfCheckGPTDetector` uses the base model for sentence embeddings by default. Optionally install `sentence-transformers` and pass a model name for better quality.
- Qwen2.5-1.5B results are illustrative only; the model is too small to reliably act on uncertainty signals.

## Reproduction Checklist

- [ ] Python environment created and dependencies installed
- [ ] Model downloaded to expected path
- [ ] HaluEval data present at `data/halueval/data.jsonl`
- [ ] `pytest tests/ -q` passes
- [ ] Small benchmark runs without errors
- [ ] Output JSON contains `summary` with accuracy, CIs, and p-values
- [ ] For 4090: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set

## Contact / Issues

Open an issue on GitHub or contact the author for reproduction problems.
