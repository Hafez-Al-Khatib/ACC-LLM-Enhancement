# Comprehensive Code Audit Report

**Date:** 2026-06-07
**Scope:** `src/acc_conflict_detector.py`, `src/acc_layer.py`, `src/acc_integration.py`, `src/halueval_detector.py`, `src/acc_intervention.py`, `src/baselines.py`, `scripts/evaluate_all_methods.py`
**Auditor:** Kimi Code CLI

---

## Executive Summary

The codebase implements an interesting neuroscience-inspired hallucination-detection framework, but it suffers from **critical logical errors**, **broken synthetic logits**, **gradient/memory leaks**, **fundamentally unfair evaluation comparisons**, and **missing cleanup**. Several components appear to have been merged from different prototypes without integration testing, leading to API mismatches and silently incorrect behavior.

**Top 5 Critical Issues**
1. `HaluEvalDetector.forward` synthetic logits are mathematically inverted — "supported" always wins (CRITICAL)
2. `acc_integration.py` post-hoc conflict detection re-runs `generate()` with sampling, producing hidden states that do **not** match the returned sequence (CRITICAL)
3. `PredictiveCodingDetector.get_layer_contributions` accumulates gradients on every call without zeroing, causing memory leaks and corrupt attributions (CRITICAL)
4. `evaluate_all_methods.py` unfair comparison — baseline uses top-p=0.95 while detectors use pure softmax sampling (HIGH)
5. `evaluate_all_methods.py` trains SAPLMA on the evaluation set and then tests on the same samples (data leakage) (HIGH)

---

## File-by-File Analysis

### 1. `src/acc_conflict_detector.py`

#### Critical
- **Gradient accumulation leak in `get_layer_contributions` (lines 392-451)**
  - The method calls `.backward()` three times inside `explain()` but never wraps calls in `torch.no_grad()`, nor does it zero parameter gradients between calls. Gradients accumulate on `primary_head`, `secondary_head`, `conflict_score_head`, and all predictor MLPs. Repeated calls to `explain()` will eventually OOM or silently corrupt downstream training.
  - **Fix:** Wrap the entire body in `with torch.no_grad():` and use `torch.autograd.grad` instead of `.backward()` if gradients are truly needed.

#### High
- **Missing hook cleanup / memory leak in `MultiLayerGenerationExtractor` (lines 545-693)**
  - Hooks are registered in `__init__` but there is no `__del__` or context-manager protocol. If an extractor is created and discarded without explicit `remove_hooks()`, the model retains stale forward hooks, leaking memory and potentially firing on unrelated generation calls.
  - **Fix:** Implement `__del__` and/or `__enter__`/`__exit__` for context-manager usage. Add `try/finally` in all callers.
- **Inconsistent architecture support (lines 575-599)**
  - `_count_model_layers` checks `model.model.decoder.layers` (OPT), but `_get_target_layer` does **not**, causing a mismatch where layer counting succeeds but hook registration fails for OPT models.
  - **Fix:** Unify architecture resolution into a single helper.

#### Medium
- **Synchronous CPU-GPU transfer bottleneck in hooks (line 609)**
  - `output[:, -1:, :].detach().cpu()` in the forward hook blocks the GPU on every layer, every step. For 4 layers and 50 tokens this is ~200 blocking transfers.
  - **Fix:** Keep buffers on GPU and move to CPU only in `get_records()` (or use pinned-memory async copies).
- **`batch_size > 1` silently unsupported despite hooks capturing all batches (line 652)**
  - Hooks append tensors with the full batch dimension, but `get_records` hard-codes batch index 0 and raises for `batch_size != 1`. The mismatch between what hooks store and what the API allows is confusing.
  - **Fix:** Either support full batching or slice the batch in the hook to avoid wasted memory.
- **`batch_size` inference in `HierarchicalPredictionErrorModule.forward` is fragile (lines 108-113)**
  - Iterates `hidden_states.values()` to get `batch_size`; if the dict is empty it raises. Could simply use `next(iter(hidden_states.values())).shape[0]`.

#### Low
- **`ALL_LABELS` concatenates primary + secondary into a flat 5-class space (line 168)**
  - The semantics are nonsensical ("hallucinated" is not mutually exclusive with "unsupported"). Kept for backward compatibility, but should be deprecated.
- **`HiddenStateExtractor.get_states` is a destructive read (line 784-787)**
  - Returns states and clears the internal list. Surprising side-effect; callers cannot inspect states twice.

---

### 2. `src/acc_layer.py`

#### High
- **`EntropyMonitor` event list is not batch-aware**
  - Events are appended without a batch index. When used in `_ACCLogitsProcessor` with `batch_size > 1`, events from all sequences are interleaved and then replicated to every item in `ACCGenerationOutput` (see `acc_integration.py` analysis).
  - **Fix:** Add `batch_idx` to `EntropyEvent` or maintain per-batch event lists.

#### Medium
- **`SlidingWindowEntropy.percentile` sorts on every call (line 91)**
  - `O(n log n)` for every percentile query. With `window_size=32` this is negligible, but for larger windows a `SortedList` or `numpy.percentile` would be cleaner.
- **`EntropyMonitor.compute_entropy` accepts 3D logits but docstring says "standard generation convention" (line 179)**
  - For 3D input it silently uses `logits[0, -1]`. If the caller passes `(batch, seq, vocab)` with `batch > 1`, the other batches are ignored without warning.

#### Low
- **Docstring mismatch on `Action` validation (line 152)**
  - Docstring says valid actions are `{"flag", "regenerate", "warning"}` but code also accepts `"suppress"`.
- **`EntropyMonitor.calibrate` does not create `results/` directory if missing**
  - Not applicable to this file directly, but `calibrate` writes nothing; the caller in `evaluate_all_methods.py` does. Minor note.

---

### 3. `src/acc_integration.py`

#### Critical
- **Post-hoc detection analyzes a DIFFERENT generation than the one returned (lines 795-864)**
  - When `use_realtime_conflict_detector=False` but `use_conflict_detector=True`, the code runs `model.generate()` a **second time** with `do_sample=True` (or whatever was passed) and attaches the extractor. Because sampling is stochastic, the second run produces **different tokens**. The hidden-state buffers therefore do **not** correspond to `sequences` returned from the first run. The conflict labels are effectively random with respect to the actual output text.
  - **Fix:** Cache hidden states during the main generation pass, or force `do_sample=False` and identical logits-processor behavior in the second pass. Better yet, eliminate the second generation entirely.

#### High
- **Real-time "regenerate" multiplies logits instead of dividing (lines 391-398)**
  - `next_scores[b] = row_logits * multiplier` makes the distribution **sharper** (lower effective temperature). The docstring says `effective_temperature = base_temperature / multiplier`, which would also make it sharper. This is the opposite of typical "regeneration" semantics (increase temperature for diversity).
  - **Fix:** Decide on the desired semantics. If regeneration should increase diversity, use `next_scores[b] = row_logits / multiplier`.
- **No try/finally around hook registration in `generate()` (lines 751-873)**
  - If `model.generate()` raises an exception after hooks are registered, `self._conflict_extractor.remove_hooks()` is never called, leaking hooks onto the model.
  - **Fix:** Wrap the generation + post-processing in `try/finally`.
- **Batch-level metrics are computed from a shared, interleaved monitor (line 947)**
  - `confidence_score=[self.monitor.get_confidence_score()] * batch_size` gives every batch item the same score derived from entropies across ALL batch items. The sliding window is polluted by cross-sequence tokens.
  - **Fix:** Maintain one `EntropyMonitor` per batch item, or compute confidence post-hoc from `per_token_entropy`.

#### Medium
- **Token-based text splicing can corrupt Unicode/subword boundaries in `_decode_with_markers` (lines 1029-1054)**
  - `tokenizer.decode(seq[cursor:pos])` + marker + `tokenizer.decode(seq[pos:])` is not guaranteed to equal decoding the full sequence with markers inserted, because byte-level BPE tokens can split multi-byte characters across the boundary.
  - **Fix:** Decode the full sequence to a string and insert markers at character offsets (requires tracking token-to-char mapping).
- **`gen_step > 0` skips the first generated token for real-time detection (line 348)**
  - The first generated token is never classified. The comment says "skip first step: no generated token yet", but the hidden-state buffer already contains the prompt encoding at index 0, and the first generated token at index 1. The logic is confusing and causes an off-by-one omission.
- **`events` field in `ACCGenerationOutput` replicates global events for every batch item (line 945)**
  - All batch items receive the identical, complete event list. Misleading for downstream consumers.

#### Low
- **`_ACCLogitsProcessor` stores `self.prompt_len` on first call only (line 321)**
  - Assumes all batch items have identical prompt length. True for standard generation, but not enforced.
- **`return_dict_in_generate=False` returns raw `sequences` but all per-token tracking is lost to the caller (line 931-932)**
  - The tracking lists are built but immediately discarded. Either always return the rich output or warn the caller.

---

### 4. `src/halueval_detector.py`

#### Critical
- **Synthetic logits are mathematically inverted — primary argmax is always "supported" (lines 116-119)**
  - For any `prob < 1.0`, `primary_logits[:, 0] = -log(prob)` is **positive**, while `primary_logits[:, 1] = log(prob)` is **negative**. Therefore "supported" (index 0) almost always wins, even when `prob = 0.99` (high hallucination probability).
  - Secondary logits have the same problem: `secondary_logits[:, 1] = -log(prob)` is always positive, so "contradictory" (index 1) always wins over "hallucinated".
  - **Fix:** Use `log(1 - prob)` for supported and `log(prob)` for unsupported. Add unit tests asserting expected argmax for prob=0.1, 0.5, 0.9.
- **`torch.load` without `weights_only=True` (line 63)**
  - Security risk in PyTorch 2.0+. Can execute arbitrary code from malicious checkpoints.
  - **Fix:** `torch.load(checkpoint_path, map_location=device, weights_only=True)`.

#### High
- **`prev_state` is accepted but ignored — fake temporal integration (lines 95, 128)**
  - The signature matches `PredictiveCodingDetector.forward(..., prev_state)` but `prev_state` is discarded and `next_state` is just the current conflict score. This breaks temporal chaining for any caller expecting real stateful behavior.
  - **Fix:** Either implement a leaky integrator like `PredictiveCodingDetector` or remove `prev_state` from the signature to avoid confusion.

#### Medium
- **`predict_sequence` always computes secondary label regardless of primary (line 142)**
  - `PredictiveCodingDetector.classify` only computes secondary when `primary == "unsupported"`. `HaluEvalDetector.predict_sequence` ignores this convention.
  - **Fix:** Gate secondary computation on primary result.
- **Missing `classify()` method (API inconsistency with `PredictiveCodingDetector`)**
  - Only `forward()` and `predict_sequence()` are provided. Callers using the richer `classify()` dict format must adapt.

#### Low
- **`_compute_features` duplicates hidden-state dict entries by storing both positive and negative indices when used with `acc_intervention.py` (see that file)**
  - Not a bug here, but the interaction wastes memory.

---

### 5. `src/acc_intervention.py`

#### High
- **`generate_with_intervention` returns different dict keys depending on code path (lines 155-192)**
  - When no conflict is detected, the dict contains `"avg_conflict"`. When conflict is detected and regeneration happens, `"avg_conflict"` is **omitted**. This forces callers to use `.get()` defensively.
  - **Fix:** Always return the same key set; set missing values to `None` or `0.0`.
- **`_generate_with_scores` grows `input_ids` with `torch.cat` in a loop (line 113)**
  - `O(n²)` memory copies because a new tensor is allocated on every token. For `max_new_tokens=50` this is minor; for 500+ it becomes severe.
  - **Fix:** Use a Python list to accumulate token IDs, then `torch.tensor` once, or use `torch.cat` on a pre-allocated buffer.

#### Medium
- **Top-p filter implementation uses `mask[1:] = mask[:-1].clone()` then `mask[0]=True` (lines 93-97)**
  - This is the "inclusive" top-p variant, but if `cumsum[0] > top_p` the mask still keeps the top token. This is usually desired, but the logic is subtle and uncommented.
- **`torch.manual_seed(seed)` does not set CUDA RNG state (line 62)**
  - For CUDA models, sampling will not be deterministic even with a fixed seed.
  - **Fix:** Also call `torch.cuda.manual_seed(seed)` (and `manual_seed_all` for multi-GPU).
- **`last_token_hs` stores both positive and negative indices, doubling dict size (lines 76-81)**
  - For a 28-layer model this is 56 dict entries per step, most unused.
  - **Fix:** Only store indices actually needed by `self.detector.layer_pairs`.

#### Low
- **Method naming inconsistency: `generate_simple_baseline` returns a `str`, while `generate_with_intervention` returns a `Dict`**
  - Callers must handle different return types.

---

### 6. `src/baselines.py`

#### High
- **Unfair generation comparisons: baseline uses top-p=0.95, baselines do not (general)**
  - `DoLaDetector`, `SAPLMADetector`, and `EntropyDetector` all do manual sampling with `F.softmax(logits / 0.8)` and **no top-p filtering**. The baseline in `evaluate_all_methods.py` uses top-p=0.95. Any observed differences in quality may be due to sampling strategy rather than detection capability.
  - **Fix:** Add configurable `top_p` and `temperature` to all baseline `detect_sequence` methods, or use `model.generate()` uniformly.
- **SAPLMA `train_on_examples` trains and evaluates on overlapping data when called from `evaluate_all_methods.py` (lines 143-182)**
  - The evaluation script passes the first 2 factual and first 2 hallucination prompts from `SAMPLES` as training data, then evaluates on the full `SAMPLES` list (which includes those same 4 prompts). This is textbook data leakage.
  - **Fix:** Use a held-out test set, or at least exclude training prompts from evaluation.

#### Medium
- **DoLa KL divergence can produce `nan` due to `log(0)` (line 91)**
  - `m = 0.5 * (p_prem + p_mat)` can have zero entries. `m.log()` yields `-inf`, and `F.kl_div` with `-inf` can return `nan` depending on PyTorch version.
  - **Fix:** Add epsilon before log: `(m + 1e-10).log()`.
- **Unnecessary CPU-GPU round-trip in `SAPLMADetector.detect_sequence` (line 199)**
  - `last_hidden` is moved to CPU at line 197, then back to `self.device` at line 199. Just keep it on the target device.
- **All baseline detectors reimplement `model.generate()` manually, losing KV-cache acceleration (general)**
  - For long sequences this is ~10× slower than `model.generate(use_cache=True)`.
  - **Fix:** Use `model.generate()` with `output_hidden_states=True` where possible, or manually manage the KV cache.

#### Low
- **Hardcoded temperature 0.8 in all baseline detectors (lines 96, 204, 248)**
  - Not configurable via constructor or method arguments.
- **`DoLaDetector.__init__` stores `self.device` but never uses it**
  - Tensors are kept on the model's device implicitly.
- **Inconsistent `detect_sequence` signatures across detectors**
  - `DoLaDetector.detect_sequence(self, input_ids, ...)` vs `SAPLMADetector.detect_sequence(self, model, input_ids, ...)`. The evaluation script paper-overs this with a `try/except TypeError`, which is brittle.

---

### 7. `scripts/evaluate_all_methods.py`

#### Critical
- **Fundamentally flawed detection metric computation (lines 111-140)**
  - Token-level precision/recall are computed per sample, but the definitions conflate sequence-level and token-level concepts. For factual prompts, precision is always `0.0` because `tp=0` and `fp=flagged`. For hallucination prompts, precision is always `1.0` because `fp=0`. Averaging these gives a meaningless number that does not reflect actual detector performance.
  - **Fix:** Define proper sequence-level metrics (e.g., sequence correctly flagged / not flagged) or use token-level TP/FP/FN across the entire corpus, not per-sample averages.

#### High
- **Data leakage in SAPLMA training (lines 165-167)**
  - As noted in `baselines.py`, training examples are drawn from the evaluation set.
- **Missing directory creation before writing JSON (line 274)**
  - `Path("results/unified_evaluation.json")` will crash if `results/` does not exist.
  - **Fix:** `out.parent.mkdir(parents=True, exist_ok=True)`.
- **`MODEL_NAME` is a relative path (line 32)**
  - Running the script from any directory other than the project root causes `FileNotFoundError`.
  - **Fix:** Resolve relative to `__file__` (a pattern already used for `_PROJECT_ROOT`).

#### Medium
- **`generate_with_detector` catches broad `TypeError` (lines 99-102)**
  - Any `TypeError` inside `detect_sequence` (even an unrelated bug) is silently retried with a different signature, masking the real error.
  - **Fix:** Use `inspect.signature` to dispatch explicitly, or normalize detector interfaces.
- **`HaluEvalDetector` checkpoint path is hardcoded and unchecked (line 176)**
  - If `adapters/custom_detector.pt` is missing, the script crashes with `FileNotFoundError`.
  - **Fix:** Check `Path(checkpoint_path).exists()` and skip the method with a warning if missing.
- **Sample size is far too small for meaningful statistics (10 samples)**
  - Results are anecdotal, not empirical. Acceptable for a smoke-test, but should be labeled as such.

#### Low
- **`judge` function uses overly broad substring matching (lines 56-61)**
  - `"not"` matches "nothing", "note", "notable", etc. `"as an ai"` is case-sensitive but `clean` is lower-cased, so it will never match (the literal has lowercase "ai" but the check is `any(p in clean for p in markers)`; wait, `"as an ai"` is all lowercase, so it WILL match). Still, the heuristic is extremely weak.
- **Hardcoded constants scattered throughout (MAX_NEW_TOKENS=12, SEED=42, DEVICE="cpu")**
  - No CLI args or config file support.

---

## Prioritized Remediation Roadmap

| Priority | Severity | File | Issue | Recommended Fix |
|----------|----------|------|-------|-----------------|
| 1 | **Critical** | `src/halueval_detector.py` | Synthetic primary/secondary logits are inverted; "supported" always wins | Replace with `log(1-prob)` / `log(prob)` mapping; add unit tests for argmax at prob=0.1, 0.5, 0.9 |
| 2 | **Critical** | `src/acc_integration.py` | Post-hoc detection re-runs sampling, producing mismatched hidden states | Cache hidden states during main generation; remove second `model.generate()` call |
| 3 | **Critical** | `src/acc_conflict_detector.py` | `get_layer_contributions` accumulates gradients without zeroing | Wrap in `torch.no_grad()`; use `torch.autograd.grad` with `create_graph=False` |
| 4 | **High** | `src/acc_integration.py` | No `try/finally` for hook cleanup | Wrap generation + post-processing in `try/finally`; ensure `remove_hooks()` always runs |
| 5 | **High** | `scripts/evaluate_all_methods.py` | Unfair sampling: baseline uses top-p, detectors do not | Add top-p filtering to all baseline `detect_sequence` methods or use `model.generate()` uniformly |
| 6 | **High** | `scripts/evaluate_all_methods.py` | SAPLMA trained on evaluation set (data leakage) | Split `SAMPLES` into train/test; exclude train samples from evaluation metrics |
| 7 | **High** | `src/acc_integration.py` | Regeneration multiplies logits (sharper, not more diverse) | Clarify semantics; if diversity is desired, divide logits by multiplier |
| 8 | **High** | `src/acc_integration.py` | Batch-level confidence score computed from interleaved window | Maintain per-batch entropy lists; compute confidence per sequence post-hoc |
| 9 | **High** | `src/acc_intervention.py` | Return dict keys vary by code path | Standardize return dict; include all keys with defaults |
| 10 | **High** | `src/acc_conflict_detector.py` | Stale hooks if extractor is garbage-collected | Implement `__del__` and context-manager support |
| 11 | **Medium** | `src/acc_conflict_detector.py` | Blocking `.cpu()` in forward hooks | Move to GPU buffers; defer CPU transfer to `get_records()` |
| 12 | **Medium** | `src/acc_integration.py` | Token-based marker splicing corrupts Unicode | Use token-to-char offset mapping before inserting markers |
| 13 | **Medium** | `src/acc_intervention.py` | `torch.cat` loop over tokens is O(n²) | Accumulate token IDs in a Python list; convert to tensor once |
| 14 | **Medium** | `src/baselines.py` | Manual generation loops lack KV-cache | Refactor to use `model.generate(output_hidden_states=True)` where feasible |
| 15 | **Medium** | `src/baselines.py` | DoLa KL divergence can `nan` | Add epsilon before `log()` in Jensen-Shannon computation |
| 16 | **Medium** | `src/halueval_detector.py` | `torch.load` without `weights_only=True` | Add `weights_only=True` |
| 17 | **Medium** | `scripts/evaluate_all_methods.py` | Metrics definitions are nonsensical | Redefine as sequence-level accuracy or corpus-level token metrics |
| 18 | **Medium** | `scripts/evaluate_all_methods.py` | Missing `results/` directory guard | `mkdir(parents=True, exist_ok=True)` before writing JSON |
| 19 | **Low** | `src/acc_conflict_detector.py` | `HiddenStateExtractor.get_states` clears internal buffer | Rename to `pop_states()` or document destructive behavior |
| 20 | **Low** | General | Hardcoded temperatures / thresholds | Centralize constants in a config dataclass or YAML file |

---

## Architectural Concerns

1. **Tight coupling between generation and detection**
   - Several detectors reimplement their own generation loop just to access hidden states. This fragments sampling logic and makes fair comparison impossible. A unified `generate()` wrapper that returns hidden states alongside sequences would eliminate this duplication.

2. **Temporal state handling is inconsistent**
   - `PredictiveCodingDetector` has a proper leaky integrator. `HaluEvalDetector` accepts `prev_state` but ignores it. The intervention engine doesn't use temporal state at all. This suggests the stateful API was bolted on after the fact without updating all implementations.

3. **Post-hoc vs real-time duality is broken**
   - The post-hoc path in `ACCEnhancedGenerator.generate()` is not just slow—it is **wrong** because it analyzes a different stochastic sample. The architecture should either (a) always extract hidden states in real time, or (b) force deterministic decoding for post-hoc analysis.

4. **No unit tests for core mathematical invariants**
   - There are no tests asserting that:
     - `HaluEvalDetector` primary argmax flips from supported→unsupported at the intended probability threshold
     - `get_layer_contributions` gradients sum to something reasonable
     - `MultiLayerGenerationExtractor` buffer indices align 1:1 with generated tokens
   - Adding these would have caught the critical logits inversion and the post-hoc mismatch.

---

## Conclusion

The codebase contains genuine innovation (hierarchical predictive coding, unified decision engine) but is currently **unsafe for production or publication** without addressing the critical bugs above. The top three blockers are:

1. **Fix `HaluEvalDetector` synthetic logits** — without this, the detector is effectively a no-op.
2. **Fix post-hoc hidden-state caching** — without this, conflict labels are random.
3. **Fix gradient accumulation in `get_layer_contributions`** — without this, long-running inference will OOM or corrupt model weights.

After these are resolved, the evaluation script should be rewritten to use a fair, shared generation backend and proper train/test splits.
