#!/usr/bin/env bash
# Reproduce the full ACC benchmark pipeline.
# Usage: bash scripts/reproduce.sh [model_name_or_path]

set -e

MODEL="${1:-Qwen/Qwen2.5-7B}"
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "ACC Reproduction Pipeline"
echo "Model: $MODEL"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

# 1. Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Detect CUDA vs CPU
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA detected. Installing CUDA PyTorch..."
    pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu124
fi
pip install -r requirements.txt

# 2. Unit tests
echo ""
echo "Running unit tests..."
pytest tests/ -q

# 3. Download model if missing
if [ ! -d "models/$MODEL_SAFE" ]; then
    echo ""
    echo "Downloading model $MODEL ..."
    python scripts/download_model_robust_hf.py "$MODEL" \
        --local-dir "models/$MODEL_SAFE" \
        --max-retries 20
fi

# 4. Run benchmark evaluation
OUTPUT="results/benchmark_eval_${MODEL_SAFE}_repro.json"
echo ""
echo "Running benchmark evaluation -> $OUTPUT"
python scripts/run_benchmark_eval.py \
    --model "models/$MODEL_SAFE" \
    --halueval 200 \
    --truthfulqa 200 \
    --pubmedqa 100 \
    --max-new-tokens 30 \
    --seed 42 \
    --use-llm-judge \
    --judge-type openai \
    --openai-model gpt-4o-mini \
    --output "$OUTPUT"

# 5. Generate ablation if on a model we can fit locally (skip for 7B)
if [[ "$MODEL_SAFE" == *"1.5B"* ]] || [[ "$MODEL_SAFE" == *"1.5b"* ]]; then
    echo ""
    echo "Running ablation study..."
    python scripts/run_ablation.py \
        --model "models/$MODEL_SAFE" \
        --max-new-tokens 15 \
        --seed 42 \
        --output "results/ablation_study_${MODEL_SAFE}_repro.json"
fi

echo ""
echo "=========================================="
echo "Reproduction complete."
echo "Main results: $OUTPUT"
echo "=========================================="
