#!/usr/bin/env bash
# Run publication-scale benchmark evaluation on RTX 4090.
# Usage: bash scripts/run_4090_benchmark.sh [model_name_or_path]

set -e

MODEL="${1:-Qwen/Qwen2.5-7B}"
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "4090 Benchmark Evaluation"
echo "Model: $MODEL"
echo "=========================================="

# Install deps and download model if not present
if [ ! -d "models/$MODEL_SAFE" ]; then
    echo "Model not found locally. Running setup..."
    python scripts/setup_4090.py --model "$MODEL"
else
    echo "Model found at models/$MODEL_SAFE"
fi

# Run benchmark evaluation
python scripts/run_benchmark_eval.py \
    --model "models/$MODEL_SAFE" \
    --halueval 200 \
    --truthfulqa 200 \
    --pubmedqa 100 \
    --max-new-tokens 30 \
    --use-llm-judge \
    --output "results/benchmark_eval_${MODEL_SAFE}_500samples.json"

echo "=========================================="
echo "Evaluation complete. Results saved to:"
echo "results/benchmark_eval_${MODEL_SAFE}_500samples.json"
echo "=========================================="
