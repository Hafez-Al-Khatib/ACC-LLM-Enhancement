# ACC LLM Enhancement — Research Engineering Audit Report
**Auditor:** Senior Research Engineer (Sub-agent)  
**Date:** 2026-05-30  
**Scope:** Full codebase review (`src/`, `scripts/`, `experiments/`, `configs/`, `paper/draft.md`)  
**Severity Legend:** 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low

---

## 1. Executive Summary

| Category | Count | Summary |
|----------|-------|---------|
| 🔴 Critical Blockers | 8 | Experiments will crash, produce nonsense, or fail to replicate paper claims |
| 🟠 High Severity | 6 | Major bugs, security vulnerabilities, architectural mismatches |
| 🟡 Medium Severity | 9 | Code quality, maintainability, missing error handling |
| 🟢 Low / Design | 5 | Documentation gaps, cosmetic issues, suboptimal patterns |

**Bottom line:** The codebase is **not publication-ready**. Several core experiments cannot run as documented, the paper claims oversell what the code actually implements, and the only existing validation result (`results/acc_validation.json`) demonstrates a **completely non-functional entropy monitoring system** that flags 100% of tokens as uncertain and emits gibberish.

---

## 2. Critical Blockers (🔴)

### CB-1: Entropy Monitor Completely Broken on Tiny-GPT2 Validation
**File:** `scripts/validate_acc.py`, `configs/acc_test.yaml`, `results/acc_validation.json`  
**Issue:** The validation config uses `threshold: 3.5` with `base_model: sshleifer/tiny-gpt2`. Tiny-GPT2 has a vocabulary of only ~1,000 tokens, so maximum possible entropy is `ln(1000) ≈ 6.9` nats. A threshold of 3.5 causes **every single token** to breach, producing outputs where `[UNCERTAIN]` is inserted after **every** generated token (see `results/acc_validation.json`: 640 breaches across 10 prompts, threshold_hit_rate = 1.0). The generated text is pure gibberish.  
**Impact:** The only "result" in the repo proves the system does not work. Entropy is not normalized by vocabulary size, making thresholds non-transferable across models.  
**Fix:** Normalize entropy by `log(vocab_size)` or use percentile-based calibration exclusively.

### CB-2: Conflict Detector Architecture Mismatch — Paper Claims vs. Reality
**Files:** `src/acc_conflict_detector.py`, `scripts/train_conflict_detector.py`, `scripts/generate_conflict_data.py`  
**Issue:** The paper (Section 6) and `src/acc_conflict_detector.py` describe a sophisticated `PredictiveCodingDetector` with hierarchical prediction errors, leaky temporal integration, and multi-layer classification. However:
- `scripts/train_conflict_detector.py` trains the **legacy** `LatentConflictDetector` (a simple 2-layer MLP on a *single* hidden state).
- `scripts/generate_conflict_data.py` uses the legacy `GenerationHiddenStateExtractor` (single layer, not multi-layer).
- There is **no training script** for the `PredictiveCodingDetector`.
**Impact:** Approach B as described in the paper is **untrainable** with the current codebase. The "neuroscience-backed" architecture is dead code.  
**Fix:** Write a training script for `PredictiveCodingDetector` using `MultiLayerGenerationExtractor`, or rewrite the paper to match the simple MLP actually implemented.

### CB-3: `run_full_pipeline.py` Calls Non-Existent CLI Arguments
**File:** `scripts/run_full_pipeline.py` (lines 165-172)  
**Issue:** The pipeline invokes:
```bash
python scripts/generate_conflict_data.py ... --num_samples_per_class 500
```
but the actual argument in `generate_conflict_data.py` is `--tokens_per_class`. This causes an `error: unrecognized arguments` crash.  
**Impact:** The full automation pipeline is broken at Step 5.  
**Fix:** Change to `--tokens_per_class` or update the argparse definition.

### CB-4: `compare_baselines.py` — Conflict Detector Strategy is a No-Op
**File:** `experiments/compare_baselines.py` (lines 133-162)  
**Issue:** The `generate_acc_conflict_detector` function loads the detector but then performs **standard generation** (`output_hidden_states=True` is passed to `model.generate()`, but hidden states are not available from `.generate()` in this form in standard HF). It returns `"detector_conflict_score": None` for every sample.  
**Impact:** The "ACC-ConflictDetector" baseline in comparisons is identical to Base generation. The evaluation is fraudulent.  
**Fix:** Integrate `MultiLayerGenerationExtractor` + `PredictiveCodingDetector` into the generation loop, or remove this baseline.

### CB-5: `create_tiny_model.py` Produces a Broken HF Model
**File:** `scripts/create_tiny_model.py`  
**Issue:** `TinyGPT2.forward()` returns `type('Output', (), {'logits': logits, 'loss': None})()` — a plain Python object, not a `ModelOutput` or `CausalLMOutputWithCrossAttentions`. This breaks:
- PEFT/LoRA attachment (expects proper output objects)
- `output_hidden_states=True` (ignored)
- Gradient checkpointing
- Any HF `Trainer` integration
**Impact:** The "tiny model" pipeline verification path is non-functional for anything beyond basic forward passes.  
**Fix:** Return proper `CausalLMOutputWithCrossAttentions` from `forward()`.

### CB-6: `fix_pubmedqa_format.py` Uses `eval()` on Untrusted Data
**File:** `scripts/fix_pubmedqa_format.py` (line 19)  
**Issue:** `ctx_dict = eval(match.group(1))` executes arbitrary Python code extracted from dataset files. This is a **remote code execution vulnerability**.  
**Impact:** Maliciously crafted dataset files can execute arbitrary code.  
**Fix:** Use `ast.literal_eval` or `json.loads`.

### CB-7: Attention Mask `.to()` Crash on None
**File:** `src/acc_integration.py` (line 360)  
**Issue:** `attention_mask.to(self.device) if attention_mask is not None else None` — wait, this is actually correct. Let me re-check... Actually it says `attention_mask.to(self.device) if attention_mask is not None else None` which is fine. But in `generate_from_prompt()`:
```python
return self.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc.get("attention_mask"),  # may be None
    ...
)
```
Then in `generate()`:
```python
attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
```
This is actually correct. Scratch CB-7.

**Replacement CB-7:** `download_with_token.py` contains hardcoded HF token placeholder.
**File:** `scripts/download_with_token.py`  
**Issue:** Contains `os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'` and `login(token='YOUR_HF_TOKEN_HERE')`. While currently a placeholder, this file encourages committing real tokens to git and should be deleted.  
**Impact:** Security anti-pattern; risk of accidental credential leak.  
**Fix:** Delete file; use environment variables or `huggingface-cli login` exclusively.

### CB-8: Datasets Referenced in Configs Do Not Exist
**File:** Multiple configs  
**Issue:** `configs/tiny_test.yaml` references `data/synthetic/train.jsonl` which is only created by `download_pubmedqa.py` as a fallback. `configs/acc_test.yaml` references `data/medical/train.jsonl` which may not exist. The `desktop_qlora_combined.yaml` references `experiments/datasets/combined/train.jsonl` but there is no script that creates the "combined" dataset.  
**Impact:** Training will fail with `FileNotFoundError` unless users manually run the right sequence of scripts.  
**Fix:** Add a dataset preparation script that creates all required files, or add existence checks with clear error messages.

---

## 3. High Severity Issues (🟠)

### H-1: `generate_conflict_data.py` Heuristic Labeling is Circular and Unreliable
**File:** `scripts/generate_conflict_data.py`  
**Issue:** The script generates labels by checking if the model's own output contains expected substrings. If the model changes (different adapter, temperature, sampling), the labels change. There is no ground-truth oracle. The "supported" class relies on the model generating the expected answer, which is not guaranteed.  
**Impact:** Training data quality is unreliable. The conflict detector learns the model's own biases, not an objective truth signal.  
**Fix:** Use a stronger model (e.g., GPT-4) as a labeling oracle, or manually curate a small high-quality dataset.

### H-2: Self-Consistency Checker Embeddings Use Unmerged Adapter
**File:** `src/acc_integration.py` (lines 141-153)  
**Issue:** `SelfConsistencyChecker` uses `self.model` directly for embedding extraction. If the model passed in is a `PeftModel` with an adapter, the embeddings reflect the adapter-tuned representations. However, if the adapter was merged for inference (`merge_and_unload()`), this is fine. The `infer.py` script does merge, but other scripts (`validate_acc.py`) also merge. This is actually mostly okay, but inconsistent with `run_full_pipeline.py` which doesn't show merging.  
**Downgrade to Medium.**

**Replacement H-2:** `train.py` — Missing Data Collator Causes Silent Issues
**File:** `scripts/train.py` (line 213)  
**Issue:** The `Trainer` is instantiated without a `data_collator`. Since the dataset is pre-tokenized with `padding="max_length"` and labels are set to `-100` for padding, this happens to work because all sequences are the same length. However, if `padding="max_length"` is removed or changed to dynamic padding, the Trainer will crash because PyTorch's default collator cannot batch variable-length tensors properly without a pad token strategy. The comment "No custom collator needed" is misleading.  
**Impact:** Fragile training setup; any config change breaks it silently.  
**Fix:** Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)`.

### H-3: `monitor_download.py` and `auto_launch_training.py` Use Wrong Shard Sizes
**Files:** `scripts/monitor_download.py`, `scripts/auto_launch_training.py`, `scripts/overnight.py`  
**Issue:** Each script hardcodes different expected shard sizes:
- `monitor_download.py`: ~4.9GB, ~30MB, ~8.9GB
- `auto_launch_training.py`: 4,949,453,792; 33,415,168; 9,513,178,144
- `overnight.py`: same as auto_launch
But `models/mistral_7b/` contains `.cache/huggingface/download/` with **incomplete** `.incomplete` files. The scripts do not robustly check for incomplete downloads using HF's own mechanisms.  
**Impact:** False positives in "ready" detection; training may start on corrupted shards.  
**Fix:** Use `huggingface_hub` library's integrity verification instead of manual size checks.

### H-4: `EntropyMonitor.compute_entropy` Wrongly Handles 3D Logits
**File:** `src/acc_layer.py` (lines 169-193)  
**Issue:** The docstring says it accepts `(batch, seq, vocab)`, but inside `_EntropyLogitsProcessor`, `scores` is `(batch, vocab)`. The 3D handling (`row = logits[0, -1]`) would pick the last sequence position, but in generation there is no "sequence" dimension in the scores tensor — it's just the logits for the *next* token. If a user passes 3D logits, the result is ambiguous.  
**Impact:** Potential misuse if integrated with non-standard generation loops.  
**Fix:** Restrict to 1D/2D or document that 3D is for prompt-encoding only.

### H-5: `test_prompts_with_gemma.py` References Nonexistent Model
**File:** `scripts/test_prompts_with_gemma.py`  
**Issue:** Default model is `gemma4:26b`. As of 2025-2026, Ollama does not have a `gemma4:26b` model. The latest is Gemma 2 (`gemma2:27b`).  
**Impact:** Script will fail for anyone following the default.  
**Fix:** Update to a valid model tag or make it a required argument.

### H-6: `experiments/run_ablation.py` Config Key Mismatch
**File:** `experiments/run_ablation.py` (line 46)  
**Issue:** Maps `"self_consistency_samples": ("acc", "self_consistency_samples")` but configs use `self_consistency_candidates`.  
**Impact:** Ablations on self-consistency depth will silently fail to modify the config.  
**Fix:** Align key names.

---

## 4. Medium Severity Code Quality Issues (🟡)

| ID | File | Issue |
|----|------|-------|
| M-1 | `scripts/download_pubmedqa.py` | Falls back to generic synthetic data (photosynthesis, gravity) when medical datasets fail, defeating the medical vertical purpose |
| M-2 | `scripts/validate_adapter.py` | Hardcodes `output_dir = "adapters/tiny_test"` instead of using the `adapter_dir` argument |
| M-3 | `scripts/aggregate_results.py` | `numeric_keys` extraction assumes `rows[0]` exists without checking; crashes on empty input |
| M-4 | `src/acc_conflict_detector.py` | `MultiLayerGenerationExtractor` hooks are never auto-removed after generation; repeated use leaks memory and accumulates stale hooks |
| M-5 | `scripts/train.py` | `load_model` ignores `torch_dtype` when quantization is enabled (`torch_dtype if bnb_config is None else None`) — this is correct per BNB docs but not documented in the function, confusing callers |
| M-6 | `scripts/infer.py` | `load_adapter` sets `local_files_only=False` but doesn't allow user override; can cause unexpected HF Hub downloads during inference |
| M-7 | `experiments/compare_baselines.py` | `token_level_f1` treats all tokens not in reference as hallucinated, which is overly aggressive for paraphrasing or elaboration |
| M-8 | `paper/draft.md` / `README.md` | Claim "~10 critical bugs patched" but no git history or changelog enumerates them; unverifiable |
| M-9 | `scripts/generate_conflict_data.py` | `num_layers` inference fails on models without `transformer.h` or `model.layers` (e.g., Llama-3, newer architectures) |

---

## 5. Design Concerns (🟢)

1. **Entropy is not a hallucination detector — it's a uncertainty detector.** The paper conflates high entropy with hallucination risk, but rare correct terms (e.g., "metastasize") also produce high entropy. The system will flag correct but rare medical terminology as uncertain.
2. **Self-consistency latency is unbounded.** The paper mentions N=5 candidates with max_new_tokens=50. That's 5× the inference cost. For a 7B model on an RTX 3080, this adds ~5-10 seconds per query. Not suitable for real-time clinical use as implied.
3. **Synthetic training data for Approach B lacks diversity.** The prompt banks have only 25 supported, 20 hallucinated, 25 uncertain, and 25 contradictory prompts. Heuristic labeling on such small banks cannot produce a robust classifier.
4. **No reproducibility seeding in inference scripts.** `validate_acc.py` seeds the model loading but not the generation (`do_sample=True` with fixed temperature but no `torch.manual_seed`). Results will vary across runs.
5. **Vertical coverage is aspirational.** The paper and registry list Medical, Legal, Financial, STEM, General. Only PubMedQA and SciQ datasets are actually downloadable via `auto_load.py`. Legal and Financial are "TBD" / "Pending".

---

## 6. Security Issues

| Severity | File | Issue |
|----------|------|-------|
| 🔴 Critical | `scripts/fix_pubmedqa_format.py` | `eval()` on untrusted dataset content — RCE vulnerability |
| 🟠 High | `scripts/download_with_token.py` | Hardcoded token placeholder pattern encourages credential leaks |
| 🟡 Medium | Multiple | `trust_remote_code=True` used in `auto_load.py`, `setup_model.py`, etc., without warnings about arbitrary code execution from HF Hub |
| 🟡 Medium | `scripts/download_direct.py` | Downloads over HTTP without checksum verification; MITM risk for model weights |

---

## 7. Test Coverage

**Result: ZERO tests.**

- `requirements.txt` lists `pytest`, `black`, `isort`.
- There is no `tests/` directory.
- There are no unit tests for `EntropyMonitor`, `ACCEnhancedGenerator`, `PredictiveCodingDetector`, or any training script.
- The only "validation" is `scripts/validate_acc.py`, which is an integration test that produces broken results.

---

## 8. Truth Claim Assessment: Does the Code Match the Paper?

| Paper Claim | Code Reality | Verdict |
|-------------|--------------|---------|
| "Approach B introduces a lightweight generation-time latent conflict detector... a small MLP classifier trained on hidden states of newly generated tokens" | The training script (`train_conflict_detector.py`) trains a simple MLP on **single-layer** hidden states from synthetic data. The fancy `PredictiveCodingDetector` with hierarchical prediction errors is **dead code** — no script trains or uses it. | ❌ **FALSE / MISLEADING** |
| "Both components are integrated as LogitsProcessor modules inside the standard HF generate() pipeline" | Only entropy monitoring (`_EntropyLogitsProcessor`) is integrated. The conflict detector is **not** integrated into generation in any working script (`compare_baselines.py` returns `None` for detector scores). | ❌ **FALSE** |
| "We implement and evaluate the full framework on a QLoRA-fine-tuned Mistral 7B model across medical, legal, financial, and STEM verticals" | No evidence of Mistral 7B training exists in the repo (no adapters, no logs). Only tiny-gpt2 and distilgpt2 adapters exist. Legal and financial datasets are marked "Pending". | ❌ **FALSE / UNVERIFIED** |
| "Entropy monitor computes per-token Shannon entropy... calibrated empirically on in-domain factual prompts" | The `calibrate()` method exists in `acc_layer.py` but there is **no evidence it was ever run** on Mistral 7B or any real model. The only validation result uses an uncalibrated fixed threshold (3.5) on tiny-gpt2. | ⚠️ **PARTIALLY IMPLEMENTED, NEVER VALIDATED** |
| "Self-consistency checker generates N diverse candidate continuations... and flags outlier clusters" | The `SelfConsistencyChecker` class is implemented and functional. It is used in `validate_acc.py`. However, the output in `acc_validation.json` shows `consistency_score ≈ 1.0` and `contradiction_rate = 0.0` for all prompts, including hallucination prompts, suggesting it is not discriminating effectively. | ⚠️ **IMPLEMENTED BUT INEFFECTIVE** |
| "QLoRA fine-tuning setup... rank r=32 and alpha=64 on desktop" | Config exists (`desktop_qlora.yaml`) but no trained adapter exists in `adapters/desktop_run/`. Only tiny-gpt2 test adapters exist. | ⚠️ **CONFIG EXISTS, NOT EXECUTED** |
| "Evaluation metrics: token-level hallucination F1, contradiction rate, calibration error, perplexity" | `evaluate_hallucination.py` implements crude lexical-overlap proxies for these metrics. It does not use entailment models, NLI, or any semantic similarity for hallucination detection. The metrics are toy implementations. | ⚠️ **OVERSIMPLIFIED** |

**Overall Verdict:** The paper draft significantly oversells the implementation. The code contains the *building blocks* for the described system, but critical integration pieces are missing, the most advanced component (`PredictiveCodingDetector`) is unused, and no actual evaluation on Mistral 7B has been performed.

---

## 9. Recommendations for Publication-Ready Code

### Immediate (Do Before Any Experiment)
1. **Delete or fix `scripts/download_with_token.py`** — remove hardcoded token pattern.
2. **Replace `eval()` with `ast.literal_eval` or `json.loads`** in `fix_pubmedqa_format.py`.
3. **Fix `run_full_pipeline.py`** argument `--num_samples_per_class` → `--tokens_per_class`.
4. **Normalize entropy** by `log(vocab_size)` or use only percentile-based thresholds to fix the 100% breach rate.
5. **Fix `create_tiny_model.py`** to return proper `CausalLMOutputWithCrossAttentions`.

### Short-Term (Before Running Experiments)
6. **Write a real training script for `PredictiveCodingDetector`** or remove it and rewrite the paper to match the simple MLP.
7. **Integrate the conflict detector into generation** in `ACCEnhancedGenerator` or `compare_baselines.py`. Currently it's a no-op.
8. **Add a dataset concatenation script** for `combined/train.jsonl` referenced in configs.
9. **Add a minimal test suite** (`pytest`) for `EntropyMonitor`, `SlidingWindowEntropy`, and `ACCEnhancedGenerator`.
10. **Add `DataCollatorForLanguageModeling`** to `train.py` and remove `padding="max_length"` to allow dynamic batching.

### Medium-Term (Before Paper Submission)
11. **Run the actual QLoRA training on Mistral 7B** and save checkpoints with reproducible seeds.
12. **Calibrate entropy thresholds** per vertical using the `calibrate()` method, and save the calibrated values.
13. **Use a real oracle for conflict detector labels** (e.g., GPT-4 or human annotators) instead of heuristic self-labeling.
14. **Implement proper hallucination metrics** using an NLI model (e.g., `facebook/bart-large-mnli`) rather than lexical overlap.
15. **Add `tests/` with unit tests and an integration test** that runs a full tiny-model pipeline and asserts sensible outputs.
16. **Clean up dead code:** Remove legacy `LatentConflictDetector` and `GenerationHiddenStateExtractor` if they are superseded, or clearly mark them as deprecated with migration paths.

---

*End of Audit Report*
