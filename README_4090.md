# Running Experiments on RTX 4090

These scripts are designed to be pulled from GitHub and run directly on a machine with an NVIDIA GPU (tested on RTX 4090).

## Why 4090?

- 24 GB VRAM enables **Qwen2.5-7B** in float16 (vs. 1.5B on XPU/Colab).
- Can run **all baselines + ACC** on a larger sample set.
- No file transfer needed — models and datasets download from HuggingFace.

## Quick Start

```bash
# 1. SSH into the 4090 machine and clone/pull the repo
git clone https://github.com/Hafez-Al-Khatib/ACC-LLM-Enhancement.git
cd ACC-LLM-Enhancement

# 2. Run setup (installs deps + downloads Qwen2.5-7B)
python scripts/setup_4090.py --model Qwen/Qwen2.5-7B

# 3. Run unified evaluation on 7B
python scripts/run_4090_eval.py --model models/Qwen_Qwen2.5-7B --max-new-tokens 15

# 4. Results saved to results/4090_unified_evaluation.json
```

## Available Experiments

### 1. Unified Method Comparison (recommended first run)

Compares 5 methods on 43 diverse samples (20 factual, 15 hallucination, 8 uncertain):

```bash
python scripts/run_4090_eval.py --model models/Qwen_Qwen2.5-7B --output results/4090_unified_7b.json
```

Outputs:
- `results/4090_unified_7b.json` — per-method accuracy, F1, flag rates
- Console summary with per-type breakdown

### 2. Train Model-Specific Detector

Generate custom detector data on the 4090 and train:

```bash
# Collect ~600 token-level examples from the target model
python scripts/collect_detector_data.py

# Train detector
python scripts/train_detector_custom.py

# Run evaluation with the new detector
python scripts/run_4090_eval.py --model models/Qwen_Qwen2.5-7B
```

### 3. Smaller Model Sanity Check

If 7B is too slow or OOMs, test on 1.5B first:

```bash
python scripts/setup_4090.py --model Qwen/Qwen2.5-1.5B
python scripts/run_4090_eval.py --model models/Qwen_Qwen2.5-1.5B --output results/4090_1.5b_baseline.json
```

## Expected Runtime

| Model | Setup | Evaluation (43 samples) |
|-------|-------|------------------------|
| Qwen2.5-1.5B | ~2 min | ~5 min |
| Qwen2.5-7B | ~10 min | ~25-40 min |

## Monitoring

Watch GPU usage in another terminal:

```bash
watch -n 1 nvidia-smi
```

## Transferring Results Back

Results files are small JSONs. Transfer with SCP:

```bash
scp user@4090-ip:/path/to/ACC-LLM-Enhancement/results/4090_unified_7b.json ./
```

Or commit and push from the 4090 machine:

```bash
git add results/4090_*.json
git commit -m "exp: 4090 7B evaluation results"
git push origin main
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `--max-new-tokens` to 10 or 8 |
| `huggingface_hub` errors | Run `huggingface-cli login` |
| Slow generation | Normal for 7B; ensure `device_map="auto"` is active |
| Custom detector missing | Run `collect_detector_data.py` + `train_detector_custom.py` first |

## Next Steps After Baseline

1. If 7B shows strong signal, try **Qwen2.5-14B** (may need quantization).
2. Implement **logit-shifting intervention** (stronger than phrase prepending).
3. Add **LLM-as-judge** for more reliable labels.
