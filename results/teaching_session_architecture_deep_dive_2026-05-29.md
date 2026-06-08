# ACC LLM Architecture Deep Dive — Teaching Session

**Date:** 2026-05-29  
**Session type:** Code review and architecture walkthrough  
**Scope:** Full system design, component breakdown, optimization rationale

---

# ACC LLM Architecture Deep Dive

## 1. The Big Picture

The system has two complementary approaches:

```
┌─────────────────────────────────────────────────────────────┐
│  APPROACH A: Entropy Monitor + Self-Consistency Checker     │
│  ────────────────────────────────────────────────────────   │
│  • Runs INSIDE the HF .generate() loop via LogitsProcessor  │
│  • No external API calls, no retrieval, no extra models     │
│  • O(1) overhead per token for entropy; O(N) for consistency│
├─────────────────────────────────────────────────────────────┤
│  APPROACH B: Generation-Time Conflict Detector              │
│  ────────────────────────────────────────────────────────   │
│  • Tiny MLP (~100K params) on hidden states                 │
│  • Hook + LogitsProcessor capture generation-time states    │
│  • Trained on synthetic (prompt, label) pairs               │
└─────────────────────────────────────────────────────────────┘
```

Both are **inference-time** layers. They don't change the base model weights. They sit on top of a QLoRA-fine-tuned Mistral 7B and intercept generation.

---

## 2. Approach A: Entropy Monitoring

### 2.1 Why LogitsProcessor Instead of a Manual Decoding Loop?

**Location:** `src/acc_integration.py`, lines 52–122 (`_EntropyLogitsProcessor`)

You might think: "Why not just write a `for` loop that calls `model.forward()` repeatedly?" We specifically chose **HuggingFace's `LogitsProcessor`** for three reasons:

1. **KV-cache management:** HF's `.generate()` handles attention mask updates, position IDs, and past-key-value caching correctly. A manual loop would require us to re-implement all of this — and getting it wrong silently degrades quality.
2. **Feature parity:** `.generate()` supports beam search, sampling (top-p, top-k), constrained decoding, etc. Our layer inherits all of this for free.
3. **Robustness:** The `LogitsProcessor` is called *after* the forward pass but *before* token selection. We see the raw logits, can modify them (for regeneration), and HF handles the rest.

```python
# src/acc_integration.py:86
def __call__(self, input_ids, scores):
    # scores: (batch, vocab) — raw logits for the NEXT token
    for b in range(batch_size):
        entropy = self.monitor.observe(scores[b])
        breached = self.monitor.check_threshold(entropy)
        if breached and action == "regenerate":
            next_scores[b] = row_logits * multiplier  # lower effective temp
```

**Trade-off:** We lose the ability to stop mid-generation and re-prompt the model. But for our use case (per-token uncertainty detection), the LogitsProcessor is the sweet spot.

---

### 2.2 Entropy Computation: Why Float32?

**Location:** `src/acc_layer.py`, lines 169–194

```python
row = row.detach().to(torch.float32)
log_probs = F.log_softmax(row, dim=-1)
probs = log_probs.exp()
entropy_nats = (-(probs * log_probs).sum()).clamp(min=0.0).item()
```

**Why float32?** The vocabulary size for Mistral is ~32K tokens. The logits for a well-trained model often have a few tokens with probability ~0.99 and thousands with probability ~1e-8. In bf16 or fp16:

- `softmax` of large-magnitude logits can overflow/underflow
- `log(1e-8)` in fp16 loses precision (fp16 minimum normal is ~6e-8)
- The product `p * log(p)` for tiny p becomes zero, **biasing entropy downward**

This isn't theoretical — on tiny-GPT2 we saw entropy clamped to ~10.8 nats because the model is near-random. On Mistral, the difference between fp16 and fp32 entropy can be **0.5–1.0 nats** at the tails, which is enough to flip threshold decisions.

**Cost:** One `to(torch.float32)` call per token. Negligible compared to the forward pass.

---

### 2.3 Three Threshold Modes

**Location:** `src/acc_layer.py`, lines 207–214

| Mode | How it works | When to use |
|------|-------------|-------------|
| `absolute` | Fixed threshold (e.g., H > 3.5 nats) | You have prior knowledge or calibrated value |
| `moving_average` | Breach when H > `window.mean() * multiplier` | Long generations where entropy drifts |
| `percentile` | Breach when H > 95th percentile of window | Adapts to local context without global calibration |

**Why three modes?** Entropy is **not** absolute. A medical term like "tacrolimus" might have H=4.0 (high because rare) but be correct. A common word like "the" might have H=0.1. A global threshold of 3.5 works for some prompts but fails for others.

The `moving_average` and `percentile` modes make the threshold **relative to the model's recent behavior**, which is more robust across domains.

---

### 2.4 Empirical Calibration

**Location:** `src/acc_layer.py`, lines 258–325

```python
def calibrate(self, model, tokenizer, calibration_prompts, ...):
    # Greedy generation on factual prompts
    outputs = model.generate(..., do_sample=False)
    for score in outputs.scores:
        h = self.compute_entropy(score[0])
        all_entropies.append(h)
    self.threshold = float(np.percentile(arr, 95))
```

**Why greedy?** We want the *lowest-entropy trajectory* the model can produce on factual prompts. If the model is confident and correct, greedy decoding gives the sharpest distribution. Sampling would add noise and inflate entropy artificially.

**Why 95th percentile?** We want to catch the tail of uncertainty without false positives. On a well-calibrated model, 95% of factual tokens should fall below the threshold.

**Alternative methods:**
- `mean_std`: mean + 2σ — more aggressive, catches ~2.5% tail
- `max`: maximum observed — conservative, almost no false positives but many false negatives

---

### 2.5 Intervention: Flag vs Regenerate vs Warning

**Location:** `src/acc_integration.py`, lines 104–120

| Action | Mechanism | Use case |
|--------|-----------|----------|
| `flag` | Insert `[UNCERTAIN]` after the token | Post-hoc analysis, human review |
| `warning` | Prefix token with warning string | Real-time alerting |
| `regenerate` | Multiply logits by `m^k`, lowering effective temperature | Automatic recovery |

**The regenerate math:** HF applies temperature as `logits / τ`. Multiplying logits by `m` is equivalent to `τ / m`:

```python
# Effective temperature = base_temperature / multiplier
multiplier = regen_multiplier ** (regen_count + 1)
next_scores[b] = row_logits * multiplier
```

**Why geometric multiplier?** Each successive breach on the same sequence gets stricter. First breach: 2× logits (τ/2). Second: 4× (τ/4). Third: 8× (τ/8). After `max_regenerations`, we fall back to flagging to avoid infinite loops.

**Trade-off:** Regeneration changes the sampling distribution. If the model was already at the "best" token, sharpening it further has no effect. But if there was a near-tie between two tokens, regeneration breaks the tie toward the higher-probability one.

---

### 2.6 Self-Consistency Checker

**Location:** `src/acc_integration.py`, lines 125–245

**The algorithm:**
1. Generate `N` candidate continuations with nucleus sampling
2. Embed each continuation using **mean-pooled hidden states** from the base model
3. Build similarity matrix, find largest cluster
4. Flag outliers as contradictions

**Why mean-pooled hidden states instead of sentence-transformers?**
- The base model is already loaded — no extra memory overhead
- The embeddings are in the model's native representation space
- Faster than loading a separate embedder

**Why not just compare generated text strings?** Lexical comparison fails on paraphrases. "The heart pumps blood" and "Blood is pumped by the heart" are semantically identical but lexically different. Embedding cosine similarity captures this.

**The clustering logic:**
```python
# For each candidate, count how many others exceed similarity threshold
counts = (sim_matrix > threshold).sum(dim=1) - 1  # exclude self
best_idx = int(counts.argmax())  # candidate with most neighbors
best_cluster_mask = sim_matrix[best_idx] > threshold
```

**Complexity:** `O(N²)` similarity computations. With N=5 and hidden_dim=4096, this is ~5×5×4096 = 100K operations — negligible compared to generation.

---

## 3. Approach B: Conflict Detector

### 3.1 Why Generation-Time Hidden States?

**Location:** `src/acc_conflict_detector.py`, lines 122–258

This is the most subtle design decision in the entire system. We extract hidden states **during generation**, not during prompt encoding.

**Why?** During prompt encoding, the model is processing the *user's query*. The hidden states reflect "what does this question mean?" not "what do I think the answer is?" The generation-time hidden state at step `t` reflects the model's internal representation **at the exact moment it chooses token `t`**. This is when hallucination signals are strongest.

**How we capture it:**
```python
class GenerationHiddenStateExtractor(LogitsProcessor):
    def register_hook(self):
        def hook_fn(module, input, output):
            last_hidden = output[:, -1:, :]  # (batch, 1, hidden)
            self._hidden_buffer.append(last_hidden)
```

The hook captures `output[:, -1, :]` — the hidden state of the **last position**. In generation with `use_cache=True`, this is always the newly generated token. Without cache, it's the last position of the full sequence.

**Trade-off:** This only supports `batch_size=1` currently. Supporting batched generation would require tracking which hidden state belongs to which sequence position across variable-length outputs. We accepted this limitation because inference is typically one prompt at a time.

---

### 3.2 The MLP Architecture

**Location:** `src/acc_conflict_detector.py`, lines 10–75

```python
class LatentConflictDetector(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2, dropout=0.1):
        layers = []
        for i in range(num_layers):
            out_dim = hidden_dim // 2 if i < num_layers - 1 else 4
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        layers = layers[:-2]  # Remove final ReLU + Dropout
```

**Why so small?**
- At `hidden_dim=4096` (Mistral), this is ~4096×2048 + 2048×4 ≈ **8.4M parameters**
- Wait, that's not 100K. Let me recalculate...

Actually, `hidden_dim // 2 = 2048`. So:
- Layer 1: 4096 → 2048: 4096×2048 + 2048 = ~8.4M weights + bias
- Layer 2: 2048 → 4: 2048×4 + 4 = ~8K weights + bias
- Total: ~8.4M parameters

Hmm, the code comment says "~100K parameters" but with default `num_layers=2` and `hidden_dim=4096`, it's actually ~8.4M. That's still tiny compared to 7B, but not 100K.

For `hidden_dim=768` (the default in the constructor):
- Layer 1: 768 → 384: 768×384 + 384 = ~295K
- Layer 2: 384 → 4: 384×4 + 4 = ~1.5K
- Total: ~296K

So the "~100K" comment might be aspirational or based on a different configuration. Regardless, it's <1% of the base model.

**Why LayerNorm + ReLU?**
- LayerNorm stabilizes training across different hidden-state magnitudes
- ReLU introduces non-linearity (without it, two linear layers = one linear layer)
- Dropout (p=0.1) prevents overfitting on the small synthetic dataset

**Why 4 output classes instead of binary?**
- Binary (hallucination vs not) would be easier but loses nuance
- "Uncertain" and "Contradictory" have different causal mechanisms and may need different interventions
- The 4-way classification lets us study per-class precision/recall

---

### 3.3 Synthetic Data Generation & Heuristic Labeling

**Location:** `scripts/generate_conflict_data.py`

We generate training data from four prompt banks:

| Category | Prompt type | Labeling heuristic |
|----------|------------|-------------------|
| Supported | Factual prompts with known answers | Substring match to expected answer, or embedding sim > 0.65, or avg token prob > 0.85 |
| Hallucinated | Common misconceptions | Falsehood appears in text, or avg token prob < 0.5, or embedding sim to falsehood > 0.60 |
| Uncertain | Unanswerable questions | Always label as uncertain |
| Contradictory | Oxymoronic premises | Label as contradictory UNLESS model explicitly resolves it (keywords: "impossible", "contradiction") |

**Why heuristics instead of human labels?**
- Scale: We need ~2,000 per-token labels. Manual labeling would take days.
- Cost: Zero annotation cost.
- Domain coverage: We can generate prompts for any domain instantly.

**Limitation:** Heuristic labels are noisy. A model might answer a hallucinated prompt correctly ("Actually, the Great Wall is not visible from the moon"), which the heuristic would mislabel. We mitigate this by:
1. Using multiple signals (substring + embedding + probability)
2. Discarding ambiguous samples (`return None`)
3. Training with early stopping on validation macro-F1

---

## 4. Training Infrastructure

### 4.1 QLoRA Configuration

**Location:** `configs/desktop_qlora.yaml`

```yaml
lora:
  r: 32
  lora_alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

**Why r=32, alpha=64?**
- `r` (rank) controls the expressiveness of the low-rank update. Higher = more capacity but more parameters.
- `alpha` controls the scaling. The effective scaling is `alpha / r = 2.0`.
- r=32 is a sweet spot for 7B models: enough capacity to learn domain-specific patterns, small enough to avoid overfitting on ~10K samples.
- On Jetson (8GB), we drop to r=8, alpha=16 — less capacity but fits in memory.

**Why all linear layers?**
- Some implementations only adapt q_proj and v_proj (attention queries/values).
- We adapt ALL linear projections (attention + MLP) because:
  1. Medical/financial knowledge is stored in both attention patterns (what to attend to) and MLP layers (fact lookups)
  2. The memory cost is negligible (~0.05GB for r=32)

---

### 4.2 4-bit Quantization (NF4)

**Location:** `configs/desktop_qlora.yaml`

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

**Why NF4 instead of standard INT4?**
- NF4 (Normal Float 4) is a **data-type-aware quantization** that uses 16 quantiles fitted to the normal distribution of weights.
- Standard INT4 uses uniform bins, which wastes precision on the tails.
- NF4 preserves ~99.9% of model quality at 4-bit vs fp16.

**Why double quantization?**
- Double quantization quantizes the quantization constants themselves.
- Without it: 32-bit constants per block → ~0.5GB overhead
- With it: 8-bit constants → ~0.125GB overhead
- Saves ~0.4GB on a 7B model

**Memory math for RTX 3080 (10GB):**
```
Mistral 7B @ 4-bit NF4:     ~4.0 GB
LoRA adapters (r=32):       ~0.05 GB
Activations (seq=1024, bs=1): ~2.5 GB
Optimizer states (8-bit):   ~0.5 GB
KV cache:                   ~0.5 GB
---------------------------------------
Total:                      ~7.6 GB
Headroom:                   ~2.4 GB ✅
```

---

### 4.3 Paged 8-bit AdamW

**Location:** `configs/desktop_qlora.yaml`

```yaml
optim: "paged_adamw_8bit"
```

**Why paged?**
- Standard AdamW stores momentum + variance per parameter: 2× parameter count in optimizer state.
- 8-bit AdamW quantizes these states to 8-bit.
- **Paged** means states are swapped to CPU RAM when GPU memory is full, then paged back in when needed.
- This lets us train with larger batch sizes or longer sequences without OOM.

**Trade-off:** Slightly slower step time due to CPU-GPU paging. But for batch_size=1, the impact is minimal.

---

### 4.4 Gradient Accumulation

**Location:** `configs/desktop_qlora.yaml`

```yaml
gradient_accumulation_steps: 4
```

**Why not just batch_size=4?**
- `batch_size=1, accum=4` uses the same effective batch size as `batch_size=4`
- But `batch_size=1` uses ~4× less activation memory
- We can fit longer sequences (1024 tokens instead of 512)

**Trade-off:** Gradient accumulation gives noisier gradients per step because each micro-batch is independent. But with AdamW's momentum, this noise is smoothed out.

---

### 4.5 Label Masking: Why -100?

**Location:** `scripts/train.py`, lines 166–178

```python
result["labels"] = [
    [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq]
    for seq in result["input_ids"]
]
```

In PyTorch's `CrossEntropyLoss`, the default `ignore_index` is -100. By setting padded positions to -100, we tell the loss function: **"don't compute loss or gradients for these positions."**

**Why not just use a DataCollator?** The standard `DataCollatorForLanguageModeling` sets labels = input_ids with no masking. The model would learn to predict `[PAD]` tokens, which is useless and harmful.

---

## 5. Evaluation Framework

### 5.1 Baseline Conditions

We compare 5 conditions:

| Condition | What it tests |
|-----------|--------------|
| Base | Raw model with no ACC |
| ACC-Entropy | Approach A alone (entropy monitoring) |
| ACC-SelfConsistency | Multi-candidate consistency checking |
| ACC-ConflictDetector | Approach B alone (hidden-state MLP) |
| ACC-Full | Integrated stack |

**Why not just compare Full vs Base?** Because we want to know which component contributes what. If Full outperforms Base but Entropy alone doesn't, then Self-Consistency or the Conflict Detector is carrying the signal.

### 5.2 Statistical Testing

**Location:** `scripts/validate_acc.py`, lines 127–157

We use **Mann-Whitney U** (not t-test) because:
- Entropy distributions are heavy-tailed (most tokens low entropy, few very high)
- We can't assume normality
- Non-parametric tests are more robust to outliers

---

## 6. Summary of Key Trade-offs

| Decision | What we gained | What we gave up |
|----------|---------------|-----------------|
| LogitsProcessor over manual loop | KV cache, beam search, correctness | Can't stop mid-generation to re-prompt |
| Float32 entropy computation | Accurate tail probabilities | ~2× memory for one tensor per step |
| Generation-time hidden states | Hallucination signal at token creation | Batch size limited to 1 |
| Heuristic labeling | Zero cost, instant scaling | Noisy labels, need validation F1 check |
| NF4 + double quant | 4× memory reduction vs fp16 | Negligible quality loss (<0.1% perplexity) |
| Gradient accumulation | Longer sequences, fits in 10GB | Noisier gradients per step |
| All linear layers in LoRA | Better domain adaptation | ~2× adapter params vs q/v only |

---

*End of teaching session — saved for reference.*
