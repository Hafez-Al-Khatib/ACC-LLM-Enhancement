#!/usr/bin/env bash
# Run a small validation benchmark locally on Qwen2.5-1.5B.
# This is a smoke test for the full pipeline; it does NOT use API judges.
# Usage: bash scripts/run_local_validation.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="models/qwen2.5-1.5b"
OUTPUT="results/local_validation_$(date +%Y%m%d_%H%M%S).json"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Local Validation Benchmark (1.5B)"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "=========================================="

if [ ! -d "$MODEL" ]; then
    echo "Model not found. Downloading..."
    python scripts/download_model_robust_hf.py Qwen/Qwen2.5-1.5B --local-dir "$MODEL"
fi

python scripts/run_benchmark_eval.py \
    --model "$MODEL" \
    --halueval 20 \
    --truthfulqa 20 \
    --pubmedqa 10 \
    --max-new-tokens 15 \
    --seed 42 \
    --output "$OUTPUT"

echo "=========================================="
echo "Validation complete. Results: $OUTPUT"
echo "=========================================="
