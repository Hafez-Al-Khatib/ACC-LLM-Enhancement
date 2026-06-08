# First Run Checklist — Mistral 7B Ready

**Use this checklist when `models/mistral_7b/` has all 3 model shards.**

---

## Step 1: Verify Download Complete

```bash
python scripts/validate_model_load.py
```

Expected output:
```
MODEL_LOAD_OK params=7,000,000,000-ish
```

---

## Step 2: Verify Environment

```bash
python scripts/check_environment.py
```

Expected: `READY_STATUS: green` or `yellow` (acceptable).

If CUDA is missing:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 3: Generate Calibration Prompts

```bash
python scripts/generate_calibration_prompts.py \
    --dataset experiments/datasets/pubmedqa/train.jsonl \
    --output data/calibration/pubmedqa_cal.jsonl \
    --num-prompts 100

python scripts/generate_calibration_prompts.py \
    --dataset experiments/datasets/sciq/train.jsonl \
    --output data/calibration/sciq_cal.jsonl \
    --num-prompts 100
```

---

## Step 4: QLoRA Fine-Tuning

### Option A: Single Vertical (Medical)
```bash
python scripts/train.py --config configs/desktop_qlora.yaml
```
Runtime: ~2-4 hours on RTX 3080

### Option B: Multi-Vertical Combined
```bash
python scripts/train.py --config configs/desktop_qlora_combined.yaml
```
Runtime: ~3-5 hours on RTX 3080 (12K samples)

---

## Step 5: Generate Conflict Detector Training Data

```bash
python scripts/generate_conflict_data.py \
    --model models/mistral_7b \
    --adapter adapters/desktop_run/final_adapter \
    --output data/acc_training/mistral_conflict_data.jsonl \
    --num_samples_per_class 500 \
    --max_new_tokens 20
```

**Critical:** This creates **4096-dimensional** hidden states (unlike the useless 2D tiny-GPT2 vectors).

Runtime: ~30-60 minutes

---

## Step 6: Train Conflict Detector

```bash
python scripts/train_conflict_detector.py \
    --data data/acc_training/mistral_conflict_data.jsonl \
    --save_dir adapters/acc_conflict_detector \
    --hidden_ratio 0.5 \
    --dropout 0.1 \
    --epochs 100 \
    --patience 10
```

Expected: Validation macro-F1 > 0.60 (acceptable), > 0.75 (good)

Runtime: ~5-10 minutes

---

## Step 7: Run Full ACC Validation

```bash
python scripts/validate_acc.py \
    --adapter adapters/desktop_run/final_adapter \
    --config configs/desktop_qlora.yaml \
    --dataset experiments/datasets/pubmedqa/test.jsonl \
    --output results/pubmedqa_eval.json
```

Runtime: ~30-60 minutes (depends on num_samples)

---

## Step 8: Run Ablation Experiments

```bash
python experiments/run_ablation.py \
    --vertical medical \
    --dataset pubmedqa \
    --ablate rank \
    --values 8 16 32 \
    --hardware desktop \
    --config-template configs/desktop_qlora.yaml \
    --epochs 3
```

Runtime: ~6-12 hours (3 runs × 2-4 hours each)

---

## Step 9: Aggregate Results

```bash
python scripts/aggregate_results.py \
    --input results/ \
    --output results/summary/
```

Generates:
- `summary.json` — all metrics
- `results_table.tex` — LaTeX table for paper
- `*.png` — bar plots and box plots

---

## Total Time Estimate

| Phase | Time |
|-------|------|
| Download | 2-6 hours (already in progress) |
| QLoRA training | 2-4 hours |
| Conflict data generation | 30-60 min |
| Conflict detector training | 5-10 min |
| ACC validation | 30-60 min |
| Ablations (optional) | 6-12 hours |
| **Minimum viable** | **~4-6 hours** |
| **Full sweep** | **~12-18 hours** |

---

*Print this and check off items as you go.*
