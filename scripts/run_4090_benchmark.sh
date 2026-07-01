#!/usr/bin/env bash
# Run publication-scale benchmark evaluation on RTX 4090.
# Usage: bash scripts/run_4090_benchmark.sh [model_name_or_path] [judge_type] [judge_model]
# Examples:
#   bash scripts/run_4090_benchmark.sh Qwen/Qwen2.5-7B openai gpt-4o-mini
#   bash scripts/run_4090_benchmark.sh Qwen/Qwen2.5-7B anthropic claude-3-5-sonnet-20241022

set -e

MODEL="${1:-Qwen/Qwen2.5-7B}"
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
JUDGE_TYPE="${2:-openai}"
JUDGE_MODEL="${3:-gpt-4o-mini}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/benchmark_eval_${MODEL_SAFE}_500samples_${TIMESTAMP}.json"
LOG="results/benchmark_eval_${MODEL_SAFE}_500samples_${TIMESTAMP}.log"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "4090 Benchmark Evaluation"
echo "Model: $MODEL"
echo "Judge: $JUDGE_TYPE / $JUDGE_MODEL"
echo "Output: $OUTPUT"
echo "Log: $LOG"
echo "=========================================="

# Check judge credentials
if [ "$JUDGE_TYPE" == "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it or switch to --judge-type local."
    exit 1
fi
if [ "$JUDGE_TYPE" == "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Export it or switch to --judge-type local."
    exit 1
fi

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
    --seed 42 \
    --use-llm-judge \
    --judge-type "$JUDGE_TYPE" \
    --${JUDGE_TYPE}-model "$JUDGE_MODEL" \
    --output "$OUTPUT" 2>&1 | tee "$LOG"

echo "=========================================="
echo "Evaluation complete."
echo "Results: $OUTPUT"
echo "Log: $LOG"
echo "=========================================="
