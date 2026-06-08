# Running ACC Evaluation on Google Colab

## Quick Start (5 minutes)

1. **Open the notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Upload notebook → Select `notebooks/ACC_Evaluation_Colab.ipynb`

2. **Set GPU runtime**
   - Runtime → Change runtime type → Select **T4 GPU** → Save

3. **Run cells sequentially**
   - Cell 1: Mount Google Drive (you'll be prompted to authenticate)
   - Cell 2: Clone repo & install dependencies
   - Cell 3: Download Qwen2.5-1.5B model (~3GB)
   - Cell 4: Check detector checkpoint
   - Cell 5: Run evaluation (takes ~10-15 min on T4)
   - Cell 6: Copy results to Drive

4. **Results saved to**: `MyDrive/ACC-LLM-Results/`

## What Gets Saved

| File | Description |
|------|-------------|
| `eval_log_YYYYMMDD_HHMMSS.txt` | Full console output |
| `unified_evaluation.json` | Structured results (accuracy, F1, per-sample) |
| `results_plot.png` | Bar chart comparing methods |

## Model Caching

The first run downloads the model (~3GB). It is automatically saved to:
```
MyDrive/ACC-LLM-Models/qwen2.5-1.5b/
```

Future runs copy from Drive instead of re-downloading (saves ~5 minutes).

## Running Larger Evaluations

To evaluate on more than 10 samples:

1. Edit `scripts/evaluate_all_methods.py`
2. Add more prompts to the `SAMPLES` list
3. Re-run Cell 5

Recommended: 30-50 samples for meaningful statistics.

## Running on Larger Models

Colab T4 (16GB VRAM) can handle:
- **Qwen2.5-1.5B** ✅ (current, ~3GB)
- **Qwen2.5-7B** ✅ (~14GB with float16)
- **Qwen2.5-14B** ❌ (OOM on T4)

To switch to 7B:
1. In Cell 3, change `repo_id` to `"Qwen/Qwen2.5-7B"`
2. Reduce `MAX_NEW_TOKENS` to 10 if OOM occurs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during generation | Reduce `MAX_NEW_TOKENS` to 8 or 10 |
| Model download fails | Run `!huggingface-cli login` and enter HF token |
| Drive mount fails | Re-run Cell 1 and re-authenticate |
| Slow generation | Normal on first run; model caches after that |

## Alternative: Standalone Script

Instead of the notebook, you can run:

```python
!python notebooks/colab_runner.py
```

This does the same thing in a single command.
