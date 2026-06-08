# Overnight Changelog

**Session:** 2026-05-28 dusk → 2026-05-29 dawn  
**Agent mode:** Autonomous (user asleep)

---

## Critical Bug Fixes

### 1. `requirements.txt` — Dependency Typo
- **Before:** `transforms>=2.4.0` (non-existent package)
- **After:** `torch>=2.4.0` with explicit CUDA/CPU install instructions
- **Impact:** Prevents pip install failure

### 2. `configs/acc_test.yaml` — Invalid Base Model Path
- **Before:** `base_model: "models/tiny_gpt2_safetensors"` (path does not exist)
- **After:** `base_model: "sshleifer/tiny-gpt2"` (valid HF Hub ID)
- **Impact:** ACC validation script now runs without 404 errors

### 3. `scripts/train.py` — Padding Tokens Trained Into Loss
- **Before:** `DataCollatorForLanguageModeling(mlm=False)` created labels = input_ids with no padding mask
- **After:** Pre-computed labels with `-100` mask for all padding tokens; removed collator entirely
- **Impact:** Prevents model from learning to predict `[PAD]` tokens

### 4. `src/acc_layer.py` — Entropy Regeneration Raised Temperature
- **Before:** `row_logits / multiplier` (dividing logits increases temperature → more random)
- **After:** `row_logits * multiplier` (multiplying logits decreases temperature → sharper)
- **Impact:** Regeneration now actually helps instead of hurting

### 5. Hardcoded Absolute Paths
- **Files:** `scripts/train.py`, `scripts/infer.py`, `experiments/datasets/auto_load.py`
- **Before:** `D:/ACC LLM Enhancement/...` paths scattered throughout
- **After:** All paths relative to repo root or configurable via CLI/config
- **Impact:** Code is now portable across machines

### 6. `local_files_only=True` Footgun
- **Files:** `scripts/infer.py`, `scripts/validate_acc.py`
- **Before:** `local_files_only=True` caused crashes when model/tokenizer not cached
- **After:** `local_files_only=False` (or omitted) allows HF Hub fallback
- **Impact:** First-time setup works without manual cache manipulation

### 7. WandB Crashes Training
- **Files:** `scripts/train.py`, `scripts/train_conflict_detector.py`
- **Before:** Unprotected `wandb.init()` calls crashed entire training run on missing API key
- **After:** All `wandb.init()` wrapped in try/except; `report_to=[]` in TrainingArguments
- **Impact:** Training is robust to missing WandB setup

### 8. `scripts/generate_conflict_data.py` — Layer Index Crash on Tiny Models
- **Before:** Hardcoded `layer_idx=-4` crashed on tiny-GPT2 (only 2 layers)
- **After:** Auto-clamps to `-2` when model has < 4 layers
- **Impact:** Smoke test works on any model architecture

### 9. Ground-Truth Preservation in Datasets
- **File:** `experiments/datasets/auto_load.py`
- **Before:** `text` field overwritten, losing original `instruction/input/output`
- **After:** Original fields preserved alongside formatted `text`
- **Impact:** Evaluation scripts can compare against ground truth

---

## New Files Created

| File | Purpose |
|------|---------|
| `scripts/auto_launch_training.py` | Monitors `models/mistral_7b/` for shards, auto-runs load test, auto-launches training |
| `scripts/check_environment.py` | Pre-flight checklist: CUDA, datasets, model files, dependencies |
| `scripts/aggregate_results.py` | Results aggregation, statistical tests, LaTeX tables, matplotlib plots |
| `scripts/generate_calibration_prompts.py` | Extracts calibration prompts from datasets for entropy threshold estimation |
| `scripts/quick_reference.py` | Prints all common commands and file locations |
| `results/overnight_status_report.md` | Full overnight status report |
| `results/WAKE_UP_GUIDE.md` | User-facing quick-start for when they wake up |
| `results/OVERNIGHT_CHANGELOG.md` | This file |
| `experiments/EXPERIMENTAL_PROTOCOL.md` | Detailed step-by-step experimental protocol with success criteria |
| `paper/related_work_enhanced.md` | Literature review with 15+ key papers from 2024–2025 |
| `paper/references.bib` | Complete BibTeX file with 28 citations |
| `data/acc_training/combined_conflict_data.jsonl` | Merged conflict training data (1820 records) |
| `results/acc_validation_tiny_gpt2_general.log` | Smoke test output |

---

## Experiments Run

| Experiment | Result | Notes |
|------------|--------|-------|
| Tiny GPT-2 general adapter training | **PASS** | Adapter saved to `adapters/tiny_gpt2_general/final_adapter` |
| ACC validation (tiny model) | **PASS** | Pipeline runs end-to-end; entropy always ~10.8 nats (expected for incapable model) |
| Conflict detector (tiny model) | **FAIL (expected)** | Macro-F1 = 0.12 due to 2D hidden states; will retrain on Mistral |
| Mistral 7B download | **IN PROGRESS** | 5.7 GB of ~15 GB cached; ETA 2–4 hours |

---

## Config Updates

| Config | Change |
|--------|--------|
| `configs/acc_test.yaml` | Fixed base model path |
| `configs/desktop_qlora.yaml` | Verified paths and hyperparameters |
| `configs/tiny_gpt2_test.yaml` | Verified paths and hyperparameters |

---

## Datasets Status

| Dataset | Status | Path |
|---------|--------|------|
| PubMedQA | Ready | `experiments/datasets/pubmedqa/` |
| SciQ | Ready | `experiments/datasets/sciq/` |
| General Instruction | Ready | `experiments/datasets/general_instruction/` |
| Alpaca | Failed (timeout) | — |
| FiQA | Failed (404) | — |
| Financial PhraseBank | Stalled (1 kB/s) | — |

---

## Known Limitations (Documented for User)

1. **This environment has NO GPU.** PyTorch is `2.12.0+cpu`. Training Mistral 7B here is not viable.
2. **Tiny model cannot validate detection accuracy.** 2D hidden states and near-random entropy make Approach B and threshold calibration meaningless. Real validation requires Mistral 7B.
3. **Missing financial/legal datasets.** Core experiments can start with medical + STEM.

---

*End of autonomous overnight session. Awaiting user wake-up.*
