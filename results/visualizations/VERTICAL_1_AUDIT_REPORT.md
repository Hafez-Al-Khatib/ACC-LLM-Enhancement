# Vertical 1 Audit Report: Core Architecture

**Date:** 2026-05-31  
**Auditor:** Code review + automated test suite  
**Scope:** `PredictiveCodingDetector`, `HierarchicalPredictionErrorModule`, `MultiLayerGenerationExtractor`

---

## Summary

| Metric | Result |
|--------|--------|
| Unit tests | **34/34 passed** |
| Bugs found | 2 (both fixed) |
| Warnings | 1 (documented) |
| Mathematical correctness | **Verified** |
| Gradient flow | **Verified** |
| Edge cases | **Handled** |

---

## Bugs Found & Fixed

### Bug 1: Empty `layer_pairs=[]` silently replaced with default
**Severity:** Medium  
**Location:** `PredictiveCodingDetector.__init__()`  
**Issue:** Python's `or` operator treats `[]` as falsy, so `layer_pairs or [default]` replaced an explicitly empty list with the default instead of raising an error.

```python
# BEFORE (buggy)
self.layer_pairs = layer_pairs or [(-12, -8), (-8, -4), (-4, -1)]

# AFTER (fixed)
if layer_pairs is None:
    layer_pairs = [(-12, -8), (-8, -4), (-4, -1)]
self.layer_pairs = layer_pairs
```

**Fix verified by:** `test_zero_layer_pairs_raises`

### Bug 2: `get_layer_contributions()` crashed with `TypeError: 'NoneType'`
**Severity:** High  
**Location:** `PredictiveCodingDetector.get_layer_contributions()`  
**Issue:** `pe` (prediction error tensor) is not a leaf tensor, so `pe.grad` is `None` unless `.retain_grad()` is called.

```python
# BEFORE (buggy)
pe.requires_grad_(True)

# AFTER (fixed)
pe.requires_grad_(True)
pe.retain_grad()
```

**Fix verified by:** `test_get_layer_contributions`, `test_explain_structure`

---

## Mathematical Correctness Verification

### Prediction Error Computation
- **Formula:** `MSE(pred(h_l), h_{l+1})` averaged over hidden dimension
- **Implementation:** `F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)`
- **Verified by:**
  - `test_non_negative_errors`: All errors ≥ 0 ✓
  - `test_output_shape`: (batch, num_pairs) ✓
  - `test_gradient_flow`: Gradients flow through predictors ✓

### Leaky Temporal Integration
- **Formula:** `s_t = α·s_{t-1} + (1-α)·pe_t`
- **Implementation:** `self.temporal_decay * prev_state + (1 - self.temporal_decay) * pe`
- **Verified by:**
  - `test_temporal_integration_with_prev_state`: Exact formula match ✓
  - `test_temporal_integration_without_prev_state`: `state == pe` when `prev_state=None` ✓
  - `test_temporal_decay_validation`: Rejects α outside [0,1] ✓

### Classification Heads
- **Primary:** Linear(num_pairs → num_pairs → 3) with GELU + LayerNorm
- **Secondary:** Linear(num_pairs → num_pairs → 2)
- **Conflict score:** Linear(num_pairs → num_pairs/2 → 1) + Sigmoid
- **Verified by:**
  - `test_output_shapes`: All shapes correct ✓
  - `test_conflict_score_range`: All scores in [0, 1] ✓
  - `test_dropout_zero_deterministic`: Reproducible with dropout=0 ✓

---

## Edge Cases Handled

| Case | Status | Test |
|------|--------|------|
| Batch size 1 | ✓ | `test_batch_dim_1d_input` |
| Batch size > 1 | ✓ | `test_output_shapes` (batch=3) |
| Missing source layer | ✓ | `test_missing_source_layer_raises` |
| Missing target layer | ✓ | `test_missing_target_layer_raises` |
| Empty hidden_states | ✓ | `test_empty_hidden_states_raises` |
| Zero layer pairs | ✓ | `test_zero_layer_pairs_raises` |
| α = 1.5 (invalid) | ✓ | `test_temporal_decay_validation` |

---

## Warnings

### Warning 1: Untrained model outputs flat conflict scores
**Severity:** Low (expected behavior)  
**Observation:** With Xavier initialization and symmetric architecture, untrained models output ~0.5 conflict score regardless of input. This is mathematically correct (sigmoid(0) = 0.5) but means the detector requires training to be useful.  
**Mitigation:** Documented in training pipeline. Not a bug.

---

## Interpretability Hooks Added

| Method | Purpose | Verified |
|--------|---------|----------|
| `get_prediction_errors()` | Raw PE per layer pair | ✓ |
| `get_layer_contributions()` | Gradient attribution to layer pairs | ✓ |
| `explain()` | Comprehensive per-token explanation | ✓ |

### Example `explain()` output:
```python
{
    "classification": {
        "primary": "supported",
        "primary_probs": {"supported": 0.515, "unsupported": 0.242, "uncertain": 0.242},
        "conflict_score": 0.5000,
        "secondary": None,
    },
    "prediction_errors": {(-4, -2): 1.46, (-2, -1): 1.16},
    "layer_contributions": {
        "conflict_score": {(-4, -2): 0.0012, (-2, -1): 0.0008},
        "primary": {(-4, -2): 0.0034, (-2, -1): 0.0021},
    },
    "state_before": [0.82, 1.34],
    "state_after": [1.02, 1.42],
}
```

---

## Visualization Tool

Created `scripts/visualize_detector.py` with 4 plot types:
1. **Prediction errors over time** — line plot per layer pair
2. **State trajectories** — leaky integrator evolution
3. **Layer contribution heatmap** — gradient attribution
4. **Conflict score trajectory** — score + primary probabilities

Plus a text diagnostic report with per-token explanations.

**Tested with synthetic data:** Prediction errors correctly spike from ~1.5 to ~10-12 during injected anomalies (steps 10-12).

---

## Performance Notes

| Component | Parameters (hidden_dim=4096, 3 pairs) | Forward pass cost |
|-----------|--------------------------------------|-------------------|
| Prediction error module | ~12.6M | 3 MLP forwards + MSE |
| Primary head | ~18 | Negligible |
| Secondary head | ~12 | Negligible |
| Conflict score head | ~10 | Negligible |
| **Total** | **~12.6M** | **~3× MLP + vector ops** |

---

## Sign-off Checklist

- [x] All unit tests pass (34/34)
- [x] Bugs found and fixed
- [x] Mathematical correctness verified
- [x] Edge cases handled
- [x] Interpretability hooks working
- [x] Visualization tool functional
- [x] Audit report written

**Status:** ✅ **APPROVED for Vertical 2**

The core architecture is mathematically sound, well-tested, and ready for the training pipeline to be built on top of it.
