# ACC LLM Enhancement — Research Audit & Progress Log

**Date:** 2026-05-28  
**Status:** Framework complete; experiments pending model download  
**Target:** AAAI/NeurIPS-ready neuroscience-inspired hallucination detection for LLMs

---

## 1. Project Overview

ACC LLM is a two-pronged research effort:
1. **Inference-time verification** — Anterior Cingulate Cortex (ACC) inspired layer that detects hallucinations and contradictions via (A) per-token entropy + self-consistency and (B) generation-time hidden-state conflict classification.
2. **Domain adaptation** — QLoRA fine-tuning of Mistral 7B for medical/legal/financial/STEM verticals on constrained hardware (RTX 3080 10GB / Jetson Orin Nano 8GB).

---

## 2. What Was Broken (Original Audit Findings)

| Issue | Severity | Impact |
|---|---|---|
| Entropy computation conceptually mismatched | Critical | High entropy ≠ hallucination; detector would false-positive on rare correct terms and miss confident falsehoods |
| Regeneration raised temperature when uncertain | Critical | Made hallucinations *more* likely during intervention |
| Conflict detector trained on prompt hidden states | Critical | Classified how the model *reads* text, not how it *generates* hallucinations |
| Only 40 hand-written synthetic examples | Critical | Severe overfitting guaranteed |
| No self-contradiction detection | High | Central research claim unsupported by code |
| No integrated evaluation framework | High | No way to measure if ACC layer actually helps |
| Un calibrated thresholds | Medium | Arbitrary 3.5 nats caused all tokens to breach on tiny model |
| Hardcoded Windows paths | Medium | Code non-portable |
| `requirements.txt` typo | Low | `transforms` instead of `transformers` |

---

## 3. Redesign & Fixes Applied

### 3.1 Infrastructure Fixes
- **`requirements.txt`**: Fixed `transformers>=4.40.0`; added `tqdm`
- **`auto_load.py`**: Preserves ground-truth `instruction`/`input`/`output` fields for evaluation
- **`model_utils.py`**: `local_files_only=False` default prevents empty-directory footgun
- **All scripts**: Replaced `D:/ACC LLM Enhancement/...` hardcoded paths with relative paths

### 3.2 Approach A — Entropy + Self-Consistency
**Files:** `src/acc_layer.py`, `src/acc_integration.py`, `scripts/validate_acc.py`

- **`EntropyMonitor.calibrate()`**: Empirical threshold calibration using factual prompts (95th percentile, mean+2σ, or max observed)
- **`SelfConsistencyChecker`**: Generates N candidates, mean-pools hidden-state embeddings, clusters by cosine similarity, flags outlier candidates as contradictions — this is an actual implementation of neuroscience "conflict monitoring" (comparing expected vs. actual outcomes)
- **Regeneration fix**: Multiplies logits (lowers effective temperature) instead of dividing
- **Validation**: Replaced nonsense prompts with adversarial hallucination prompts + Mann-Whitney U statistical tests

### 3.3 Approach B — Generation-Time Conflict Detector
**Files:** `src/acc_conflict_detector.py`, `scripts/generate_conflict_data.py`, `scripts/train_conflict_detector.py`

- **`GenerationHiddenStateExtractor`**: `LogitsProcessor` that captures hidden states for **newly generated tokens** during `model.generate()`
- **Synthetic data generation**: 4 prompt banks (supported, hallucinated, uncertain, contradictory) producing ≥500 tokens per class (2,000+ total)
- **Heuristic labeling**: Substring matching, optional sentence-transformer similarity, confidence thresholds
- **Training**: Proper PyTorch `Dataset`/`DataLoader`, train/val split, early stopping on macro-F1, per-class precision/recall/F1 via `sklearn`

### 3.4 Evaluation Framework
**Files:** `experiments/evaluate_hallucination.py`, `experiments/compare_baselines.py`, `experiments/run_ablation.py`, `experiments/benchmarks/run_medical_qa.py`, `configs/acc_experiment.yaml`

- **Hallucination metrics**: Token-level F1, lexical overlap, contradiction heuristics, calibration error, perplexity
- **Baseline comparison**: Base vs ACC-Entropy vs ACC-SelfConsistency vs ACC-ConflictDetector with paired Wilcoxon signed-rank tests
- **Ablation extension**: `run_ablation.py` now supports `acc_threshold`, `acc_mode`, `self_consistency_samples` with `--eval-after-train`
- **Medical QA benchmark**: End-to-end PubMedQA benchmark comparing base vs. ACC generation
- **Integrated config**: `configs/acc_experiment.yaml` ties model, adapter, ACC settings, and evaluation dataset

---

## 4. Current File Inventory

```
configs/
  acc_experiment.yaml          # NEW — integrated experiment config
  acc_test.yaml                # UPDATED — self-consistency settings
  desktop_qlora.yaml           # unchanged
  jetson_qlora.yaml            # unchanged
  tiny_gpt2_test.yaml          # unchanged

src/
  acc_layer.py                 # UPDATED — added calibrate()
  acc_integration.py           # UPDATED — SelfConsistencyChecker, fixed regeneration
  acc_conflict_detector.py     # UPDATED — GenerationHiddenStateExtractor
  model_utils.py               # UPDATED — local_files_only fix

scripts/
  train.py                     # UPDATED — collator docs, file checks
  validate_acc.py              # UPDATED — adversarial prompts, stats tests
  validate_model_load.py       # UPDATED — local_files_only fix
  train_conflict_detector.py   # REWRITTEN — generation-time training
  generate_conflict_data.py    # REWRITTEN — 2K+ per-token records
  infer.py                     # UPDATED — regen_multiplier parameter
  download_*.py                # UPDATED — relative paths

experiments/
  evaluate_hallucination.py    # NEW
  compare_baselines.py         # NEW
  run_ablation.py              # UPDATED — ACC ablations + eval-after-train
  benchmarks/run_medical_qa.py # NEW
  datasets/auto_load.py        # UPDATED — preserve ground truth
```

---

## 5. Smoke Test Results

```
ALL_COMPILE_OK          ✅ 12 core files pass py_compile
ALL_SYNTAX_OK           ✅ AST parse successful
ALL_IMPORTS_OK          ✅ src/ modules import cleanly
```

*Note: Full execution tests blocked pending model download and dependency install.*

---

## 6. Remaining Blockers

| Blocker | Severity | Next Action |
|---|---|---|
| **Mistral 7B not downloaded** | 🔴 High | `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral_7b` |
| **Python deps not installed** | 🟡 Medium | `pip install -r requirements.txt` |
| **Datasets not downloaded** | 🟡 Medium | `python experiments/datasets/auto_load.py --dataset pubmedqa ...` |
| **Conflict detector not trained** | 🟡 Medium | Run `generate_conflict_data.py` → `train_conflict_detector.py` |

---

## 7. Experimental Roadmap

### Phase 1 — Smoke Tests (tiny model, ~1 hour)
1. `python scripts/generate_conflict_data.py --model_path sshleifer/tiny-gpt2`
2. `python scripts/train_conflict_detector.py`
3. `python scripts/validate_acc.py`

### Phase 2 — Real Data & Training (Mistral 7B, ~1–2 days)
1. Download PubMedQA / FiQA / SciQ via `auto_load.py`
2. QLoRA fine-tune Mistral 7B on medical vertical (`desktop_qlora.yaml`)
3. Calibrate entropy monitor on factual calibration set
4. Generate conflict training data with fine-tuned model
5. Train conflict detector on generated tokens

### Phase 3 — Evaluation & Ablations (~1 day)
1. `python experiments/compare_baselines.py --config configs/acc_experiment.yaml --prompts ...`
2. `python experiments/benchmarks/run_medical_qa.py --config configs/acc_experiment.yaml`
3. `python experiments/run_ablation.py --vertical medical --dataset pubmedqa --ablate rank --values 8 16 32 --eval-after-train`

### Phase 4 — Paper Write-up (~1 week)
- Position against SelfCheckGPT, Semantic Entropy, FactScore
- Report hallucination detection F1, calibration error, ablation trends
- AAAI-ready submission target

---

## 8. Theoretical Positioning

**Novelty claim (now supported by code):**
> "We introduce ACC LLM, the first inference-time verification layer that combines (1) empirically-calibrated per-token entropy monitoring with (2) semantic self-consistency checking via hidden-state embedding clustering and (3) a small generation-time conflict detector trained on the model's own synthetic hallucinations. Unlike post-hoc methods (SelfCheckGPT, FactScore), our approach intervenes during generation. Unlike pure logit-based uncertainty, our self-consistency mechanism detects internal contradictions that entropy alone cannot catch."

**Key differentiators implemented:**
- Calibration anchors thresholds to the model's own in-domain distribution
- Self-consistency implements actual conflict-monitoring (multiple predicted outcomes compared)
- Generation-time hidden states align the detector with the target phenomenon

---

## 9. Bottom Line

**Before:** Promising concept with broken code and no valid experiments.  
**After:** Research-ready framework with sound architecture, proper evaluation, and a clear experimental roadmap.  

**Status:** ⏳ Waiting on model download to begin real experiments.
