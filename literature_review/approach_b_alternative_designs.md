# Approach B: Alternative Architecture Designs

**Date:** 2026-05-30  
**Purpose:** Propose 3 alternative architectures more sophisticated than the current 2-layer MLP

---

## Executive Summary

The current Approach B (2-layer MLP, 4096→2048→4) is **not competitive** with 2025 SOTA. Three alternatives are proposed below, ranked by novelty-to-complexity ratio:

| Rank | Architecture | Novelty | Complexity | Params | Key Advantage |
|------|-------------|---------|------------|--------|---------------|
| 1 | **Multi-Layer Temporal Probe (MLTP)** | High | Medium | ~25M | Captures temporal dynamics + multi-layer info |
| 2 | **Cross-Modal Attention Probe (CMAP)** | Very High | High | ~12M | Fuses hidden state + entropy + token probability via attention |
| 3 | **Contrastive Sequence Encoder (CSE)** | High | Medium | ~8M | Learns representations via contrastive learning |

---

## Alternative 1: Multi-Layer Temporal Probe (MLTP) — RECOMMENDED

### Architecture

```
Input: Sequence of hidden states from L transformer layers across T generation steps
       Shape: (T, L, hidden_dim) where T=generation length, L=number of layers tapped

Step 1: Layer Fusion
  - For each timestep t: concatenate or attention-pool hidden states from layers {-1, -4, -8}
  - Output: (T, hidden_dim * 3) or (T, hidden_dim) if attention-pooled

Step 2: Temporal Encoding
  - Bidirectional LSTM or Transformer encoder over T timesteps
  - Captures how hidden-state geometry evolves during generation
  - Output: (T, hidden_dim)

Step 3: Classification Head
  - Linear(hidden_dim → hidden_dim/2) + LayerNorm + ReLU + Dropout
  - Linear(hidden_dim/2 → 4)  [supported, hallucinated, uncertain, contradictory]
  - Output: (T, 4) logits
```

### Why It's More Novel
- **No competitor uses temporal dynamics** for hallucination detection. Single-Pass HalluDet, DCRD, and SAPLMA all use single-step hidden states.
- **Multi-layer fusion** taps information from multiple depths (surface semantics at -1, syntactic patterns at -4, conceptual representations at -8).

### Why It's More Powerful
- A hallucination often builds up over multiple tokens. Temporal modeling captures this trajectory.
- Early tokens in a hallucinated sequence may look normal; the pattern only emerges across 3-5 tokens.
- Multi-layer fusion provides richer features than any single layer.

### Implementation Complexity: MEDIUM
- Requires storing hidden states from multiple layers during generation (memory: ~3× more than single layer).
- LSTM is standard PyTorch; no exotic dependencies.
- Can reuse existing GenerationHiddenStateExtractor with multiple hooks.

### Parameter Count
- Layer fusion (attention): ~hidden_dim² = 16M
- BiLSTM (2 layers): ~2 × 4 × hidden_dim² = 32M (or use smaller hidden size)
- Classification head: ~hidden_dim × hidden_dim/2 = 8M
- **Total: ~25-50M** depending on LSTM size

### Key Risks
- **Memory:** Storing T × L × hidden_dim hidden states during training is expensive. Mitigation: process sequences in chunks or use gradient checkpointing.
- **Latency:** LSTM forward pass adds ~5-10ms per token. Acceptable for most applications.
- **Overfitting:** More parameters + small dataset = overfitting risk. Mitigation: strong regularization, smaller LSTM hidden size.

---

## Alternative 2: Cross-Modal Attention Probe (CMAP)

### Architecture

```
Inputs (3 modalities):
  1. Hidden state h_t ∈ R^hidden_dim              (from layer -4)
  2. Entropy e_t ∈ R^1                            (from entropy monitor)
  3. Token probability p_t ∈ R^1                  (max probability of generated token)

Step 1: Embedding
  - Project each modality to a common dimension d=256
  - h_t: Linear(hidden_dim → d)
  - e_t: Linear(1 → d)
  - p_t: Linear(1 → d)

Step 2: Cross-Modal Attention
  - Stack 3 embeddings: (3, d)
  - Multi-head self-attention over modalities
  - Each modality attends to the others
  - Output: (3, d) fused representation

Step 3: Pool + Classify
  - Mean-pool over 3 modality tokens → (d,)
  - Linear(d → d/2) + ReLU + Dropout
  - Linear(d/2 → 4)
```

### Why It's More Novel
- **No competitor fuses hidden states with entropy and token probability via attention.** Most probes use hidden states alone.
- The cross-modal attention learns which modality is most predictive for each token type.

### Why It's More Powerful
- Entropy provides distributional uncertainty signal that hidden states miss.
- Token probability provides confidence signal.
- Attention learns to weigh modalities dynamically: for some tokens entropy is key, for others hidden-state geometry is key.

### Implementation Complexity: HIGH
- Requires integrating entropy monitor and token probability extraction into training pipeline.
- Multi-head attention is standard but adds complexity.

### Parameter Count
- Projection layers: 3 × (input_dim × d) ≈ 3 × (4096 × 256) = 3M
- Attention: 4 × d² = 256K
- Classification head: d × d/2 = 32K
- **Total: ~3.3M**

### Key Risks
- **Integration complexity:** Must synchronize hidden-state extraction, entropy computation, and token probability logging during generation.
- **Training instability:** Three modalities with very different scales require careful normalization.

---

## Alternative 3: Contrastive Sequence Encoder (CSE)

### Architecture

```
Input: Hidden state sequence (T, hidden_dim)

Step 1: Sequence Encoder
  - Transformer encoder (2 layers, 4 heads, hidden_dim=512)
  - Processes entire generation sequence
  - Output: (T, 512)

Step 2: Contrastive Learning
  - For each token, create:
    * Anchor: its encoded representation
    * Positive: representation of a token with SAME label from another sample
    * Negative: representation of a token with DIFFERENT label
  - Triplet loss: L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

Step 3: Fine-Tune Classifier
  - After contrastive pretraining, add Linear(512 → 4)
  - Fine-tune with cross-entropy on labeled data
```

### Why It's More Novel
- **No competitor uses contrastive learning for hallucination detection probes.**
- Self-supervised pretraining on unlabeled data reduces dependence on noisy heuristic labels.

### Why It's More Powerful
- Contrastive learning learns a representation space where similar hallucination types cluster together.
- Less sensitive to heuristic label noise than direct cross-entropy training.
- Can leverage much more unlabeled data.

### Implementation Complexity: MEDIUM
- Contrastive learning is well-understood but requires careful mining of positive/negative pairs.
- Transformer encoder is standard PyTorch.

### Parameter Count
- Transformer encoder (2 layers): ~2 × (512² × 4) = 2M
- Classification head: ~512 × 4 = 2K
- **Total: ~2M**

### Key Risks
- **Data requirements:** Contrastive learning needs many more samples than supervised learning. Mitigation: generate large synthetic corpus.
- **Label ambiguity:** "Supported" tokens from different domains may not form a coherent cluster. Mitigation: use domain-aware sampling.

---

## Comparison with Competitors

| Method | Architecture | Temporal? | Multi-modal? | Training | Our Advantage |
|--------|------------|-----------|--------------|----------|---------------|
| SAPLMA | 2-layer MLP | No | No | Supervised | We have all three alternatives above |
| Single-Pass HalluDet | 3-layer MLP | No | No | Supervised | MLTP captures temporal dynamics |
| DCRD | 1-layer MLP | No | No | Supervised | All alternatives more sophisticated |
| ICR Probe | Trajectory model | Yes | No | Supervised | MLTP + CSE offer different trajectory modeling |
| **MLTP (ours)** | BiLSTM + multi-layer | Yes | No | Supervised | Only method with multi-layer + temporal |
| **CMAP (ours)** | Cross-modal attention | No | Yes | Supervised | Only method fusing hidden state + entropy + prob |
| **CSE (ours)** | Transformer + contrastive | Yes | No | Self-supervised | Only method using contrastive learning |

---

## Recommendation

**Primary: MLTP (Multi-Layer Temporal Probe)**
- Best novelty-to-complexity ratio
- Directly addresses a weakness of all competitors (they ignore temporal dynamics)
- Feasible on RTX 3080 and Jetson
- Strong biological plausibility (ACC processes sequences of prediction errors)

**Secondary enhancement: Add entropy as auxiliary input to MLTP**
- Don't build full CMAP (too complex), but concatenate entropy to LSTM input
- Minimal complexity increase, significant signal boost

---

## Integration Notes

**No modification to the 4-bit base model is required.** Hooks operate on the dequantized hidden states that flow through the model during forward pass. All probe parameters remain in full precision (bf16) and are updated via standard backprop during detector training.

**References:**
- `src/acc_conflict_detector.py` — Current Approach B implementation
- `src/acc_layer.py` — EntropyMonitor
- `src/acc_integration.py` — LogitsProcessor integration pattern
- `literature_review/COMPREHENSIVE_SYNTHESIS.md` — Competitive analysis
