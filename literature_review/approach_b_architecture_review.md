# Approach B Architecture Review: Competitive Analysis & Novelty Assessment

**Date:** 2026-05-30
**Analyst:** Senior Research Architect (sub-agent)
**Scope:** Exact architectural comparisons for the LatentConflictDetector (Approach B) vs. six direct competitors.

---

## 0. Approach B — Baseline Definition

| Attribute | Specification |
|-----------|--------------|
| **Name** | LatentConflictDetector |
| **Architecture** | 2-layer MLP |
| **Layer 1** | Linear(4096 -> 2048) + LayerNorm + ReLU + Dropout(0.1) |
| **Layer 2** | Linear(2048 -> 4) + LayerNorm |
| **Input** | Single hidden-state vector (4096-D) from generation-time forward pass (Mistral 7B) |
| **Output** | 4-way classification logits: supported, hallucinated, uncertain, contradictory |
| **Training data** | ~2,000 synthetic per-token examples with heuristic labels |
| **Inference cost** | One forward pass through the probe (~0.5-1 ms on GPU) |

**Key claims to defend:** (1) generation-time (not prompt-time) extraction; (2) per-token granularity; (3) 4-way taxonomy instead of binary; (4) heuristic synthetic labeling pipeline.

---

## 1. SAPLMA (Azaria & Mitchell, 2023)

### Exact Architecture
- **Type:** Feed-forward MLP classifier (non-linear probe).
- **Layers:** 3 hidden layers in most reproductions; the original paper described a generic feedforward network.
- **Hidden dimensions (reported in reproductions):**
  - [256, 128, 64] — most common reproduction (Azaria & Mitchell 2023; reproducers such as Skean et al. 2024).
  - [512, 256, 128] — used in some follow-up works (e.g., calibration studies).
  - [1024, 512, 256] — larger variant tested in Neural-ODE baselines (2025).
- **Activations:** ReLU after each hidden layer; sigmoid output.
- **Parameters:** ~50K-500K depending on hidden dims.

### Inputs
- **What:** Hidden-state vector from a single selected layer (typically middle-to-late, e.g., layer 20-32 of Llama 2 7B).
- **Which token:** The final token of the prompt/statement (post-hoc / statement-level). Some reproductions also average across all tokens.
- **Dimension:** Same as base-model hidden size (e.g., 4096 for Llama 2 7B).

### Output Classes
- **Binary:** P(true | h) — probability the statement is truthful.

### Training Data & Labeling
- **Size:** Several thousand statement pairs per topic.
- **Source:** Synthetic true/false statements generated from tabular data and ChatGPT; plus OPT-generated completions.
- **Labeling:** Human-verified or template-derived binary correctness labels.
- **Validation:** Leave-one-topic-out cross-validation (train on 5 topics, test on 1 held-out topic).

### Reported Performance
- **Accuracy:** 71-83% (layer-dependent, model-dependent).
- **AUROC:** ~80-90% on short-form QA in later reproductions.

### Key Limitations
1. **Prompt-level / post-hoc:** It classifies statements after they have been generated or inserted as prompts. It does not operate at generation time on per-token hidden states.
2. **Binary only:** Cannot distinguish uncertainty from contradiction.
3. **Layer sensitivity:** Performance peaks at intermediate layers and degrades if the wrong layer is chosen.
4. **Topic transfer:** While better than random, cross-topic generalization is not perfect.


---

## 2. SEPs — Semantic Entropy Probes (Kossen et al., ICML 2024 / arXiv:2406.15927)

### Exact Architecture
- **Type:** Linear probe (logistic regression classifier). **Not an MLP.**
- **Layers:** Single linear layer (optionally with a softmax/binarization step).
- **Parameters:** Effectively zero overhead beyond the linear projection.

### Inputs
- **What:** Hidden-state vector h_p^l(x) at a layer l and token position p.
- **Which token:** Typically the last token of the generated response (or the token before generation).
- **Dimension:** Model hidden size.

### Output Classes
- **Binary:** high semantic entropy vs. low semantic entropy.
  - The probe approximates Semantic Entropy (SE) — a continuous sampling-based metric — from a single hidden state.

### Training Data & Labeling
- **Size:** Training-corpus scale (thousands of prompts).
- **Labeling:** Unsupervised w.r.t. correctness. SE labels are computed by generating M stochastic samples per prompt, clustering them by NLI equivalence, and binarizing the entropy score. No ground-truth accuracy labels are required.
- **Key advantage:** Because it predicts entropy (a model-behavior property) rather than correctness, it generalizes better to out-of-distribution data than accuracy probes.

### Reported Performance
- Retains high performance for hallucination detection compared to multi-sample SE.
- Generalizes better to OOD data than SAPLMA-style correctness probes.
- Latency overhead: near zero (single forward pass).

### Key Limitations
1. **Linear capacity:** A single linear layer cannot capture non-linear interactions in the hidden state that may encode complex factual conflicts.
2. **Proxy task:** It predicts entropy, not factual correctness. A model can be confidently wrong (low entropy, high hallucination risk) or uncertain yet correct (high entropy, low risk).
3. **Training cost:** Requires M generations per training example to compute SE labels (expensive data collection).


---

## 3. Single-Pass HalluDet (NeurIPS 2025)

> **Note:** Independent verification of this exact publication title was limited during this review. The analysis below synthesizes (a) our project's competitive tracking, (b) the broader class of verified single-pass hidden-state probes (HALT, HalluShift), and (c) metrics reported in internal benchmarks.

### Exact Architecture (per project tracking)
- **Type:** 3-layer MLP.
- **Hidden dimensions:** Not publicly specified beyond 3-layer MLP on prefill state.
- **Activations:** ReLU (inferred from contemporaneous works).
- **Output:** Sigmoid (binary).

### Inputs
- **What:** Single hidden-state vector from the prefill stage (prompt encoding, before autoregressive generation begins).
- **Which token:** Last token of the prompt / prefill sequence.
- **Dimension:** Model hidden size.

### Output Classes
- **Binary:** Hallucination vs. non-hallucination.

### Training Data & Labeling
- Not independently verified. Presumed to use standard QA hallucination benchmarks (e.g., TriviaQA, NQ-Open) with binary correctness labels.

### Reported Performance (project tracking)
- **AUROC:** 0.89+
- **Latency overhead:** 10-15 ms on RTX 4090 (<1% of total generation latency).

### Key Limitations
1. **Prefill-state only:** It does not inspect hidden states during token generation; it makes a single prediction before decoding starts. This misses emergent hallucination signals that appear mid-sequence.
2. **Binary:** No gradation between uncertainty and contradiction.
3. **Verification risk:** Because the exact paper could not be retrieved, citing it in a submission without a solid bibliographic trace is risky.

### Related Verified Methods
- **HALT** (Anonymous, 2026) and **HalluShift** (Dasgupta et al., 2025) are confirmed single-pass internal-state methods achieving AUROC 0.877-0.899. They compute scalar consistency metrics (inter-layer cosine distance) rather than training an MLP classifier.


---

## 4. DCRD — Dynamic Cognitive Reconciliation Decoding (2026)

**Source:** J. Li et al., Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding, arXiv:2605.12185 (May 2026).

### Exact Architecture
- **Type:** Lightweight conflict classifier.
- **Layers:** Single-layer MLP (explicitly described as a lightweight conflict classifier (a single-layer MLP), making it resource-efficient).
- **Activations:** Not specified, but implied linear + softmax/sigmoid for routing.
- **Parameters:** Extremely small (single linear map).

### Inputs
- **What:** Attention-map features, not raw hidden states.
  - Specifically, attention relationships between the newly generated tokens and the context tokens are used to measure contextual fidelity.
- **Which tokens:** Cross-attention (or self-attention) weights between generated tokens and source context.

### Output Classes
- **Binary:** Conflict predicted (high conflict) vs. conflict-free (low conflict).
  - The output is used to route decoding: greedy decoding path vs. dynamic contrastive decoding path.

### Training Data & Labeling
- **Size:** ConflictKG benchmark (4,466 conflict / non-conflict instances) + Counterfacts + NQ-Swap.
- **Labeling:** Synthetic knowledge-conflict scenarios where context contradicts parametric knowledge. Labels are derived from answer correctness under conflict.

### Reported Performance
- **QA Accuracy gains:** +17.7% on NQ-Swap (Llama2-7B) vs. greedy decoding; +6.7% vs. CAD.
- **Inference time:** 39% of COIECD (2.4 s -> ~0.94 s per sample on average).
- **Robustness:** Performance drop of only 10.1% when conflict proportion increases, vs. 15.7% for COIECD.

### Key Limitations
1. **Domain-specific:** Targets context-memory conflicts (RAG / knowledge-update scenarios), not general hallucination or ungrounded invention.
2. **Attention-only:** Does not exploit hidden-state geometry or FFN activations, which encode factual knowledge.
3. **Binary routing:** The 4-way taxonomy of Approach B (supported / hallucinated / uncertain / contradictory) is richer than DCRD's conflict/no-conflict split.
4. **Requires contrastive decoding on conflict path:** When conflict is detected, it falls back to a more expensive decoding regime; the classifier itself is cheap, but the system latency can increase.


---

## 5. COCO — Conflict Neuron Identification (Zhang et al., 2026)

**Source:** J. Zhang et al., Modeling Implicit Conflict Monitoring Mechanisms against Stereotypes in LLMs, arXiv:2605.09647 (May 2026).

### Exact Architecture
- **Type:** Training-free, mechanistic interpretability method. Not a learned classifier.
- **Method:** Contrastive causal scoring (C2-Score) applied to individual neurons.
  - Maximizes inter-group activation difference (biased vs. unbiased outputs).
  - Minimizes intra-group activation dispersion.
  - Uses a contrastive loss (symmetric InfoNCE-style) to rank neurons.
- **No MLP.** No trainable parameters beyond the LLM itself.

### Inputs
- **What:** Per-neuron activation responses a_w^{l,j} computed by ablating (zeroing) each neuron and measuring the L2 change in the hidden state.
- **Which layers:** Attention heads across all layers.

### Output
- **Neuron set:** A ranked list of COCO neurons that causally mediate conflict monitoring.
  - No explicit classification of hallucination / contradiction / uncertainty.

### Training Data & Labeling
- **No supervised training of a detector.** Uses paired generations (biased vs. unbiased) as contrastive data.
- **Size:** Thousands of paired scenarios per social domain.

### Reported Performance
- **Causal impact:** Deactivating COCO neurons causes >90% of outputs to revert to biased content — a far stronger effect than adversarial jailbreaks (~75% bias induction).
- **Enhancement:** LE-COCO and NE-COCO editing strategies improve robustness while preserving general capability (MMLU, TruthfulQA, GPQA).

### Key Limitations
1. **Not a detector:** It identifies where conflict is encoded, but does not provide a runtime classification signal that can trigger interventions during generation.
2. **Scope gap:** Demonstrated on social stereotype debiasing; generalization to factual hallucination is claimed but not empirically validated.
3. **Dense-only:** Not evaluated on MoE or reasoning models.
4. **Scaling:** Neuron-level analysis does not scale trivially to per-token generation-time decisions.


---

## 6. ICR Probe / MultiHaluDet (2025 / 2026)

### 6a. ICR Probe (Zhang et al., 2025)
**Source:** ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs, arXiv:2507.16488 (July 2025).

#### Exact Architecture
- **Type:** MLP classifier.
- **Layers:** 4 fully connected layers: L -> 128 -> 64 -> 32 -> 1 (where L = number of LLM layers, e.g., 32).
- **Regularization:** Batch normalization + Dropout (p = 0.3) after each hidden layer.
- **Activations:** Leaky ReLU (alpha = 0.01); sigmoid output.
- **Parameters:** <16K parameters for L < 42.

#### Inputs
- **What:** Pooled ICR score vector of shape 1 x L.
  - The ICR (Internal Consistency Representation) matrix is N x L (tokens x layers), computed from hidden-state trajectory dynamics.
  - Token-wise pooling (averaging across N) yields a layer-wise summary of representational drift.
- **Dimension:** Equal to the number of layers (L).

#### Output Classes
- **Binary:** Hallucination probability.

#### Training Data & Labeling
- Annotated hallucination datasets (binary correctness labels).
- Standard supervised training (BCE loss, Adam optimizer).

#### Reported Performance
- Layer-wise AUROC peaks at middle layers (layers 15-28 for Gemma-2; layer 10 for Llama-3).
- Strong cross-dataset consistency (narrow standard deviation bands).

#### Key Limitations
- **Binary only.**
- **Requires trajectory extraction:** Needs hidden states from all layers to compute the ICR matrix, then pooling. This is more expensive than a single-vector probe.
- **Task-specific peak layers:** The best-performing layer varies by model architecture, complicating deployment.


---

### 6b. MultiHaluDet (2026)
**Source:** Multilingual Hallucination Detection via LLM Hidden State Probing, arXiv:2605.24919 (May 2026).

#### Exact Architecture
A deep, multi-stage neural architecture:
1. **Projection:** Linear -> LayerNorm -> GELU (to uniform hidden dim H).
2. **Multi-Scale Attention:** Learned position-wise gating over multiple temporal resolutions (average-pool -> project -> upsample).
3. **Layer-Weighted Transformer Encoder:** 6 Pre-LN Transformer layers (8 heads, d = 384) with a learnable layer-importance vector lambda in R^K.
4. **Self-Attention Pooling:** Two-layer MLP with tanh assigns relevance scores to each layer position.
5. **Global Branch:** Two-layer MLP on hand-crafted global features (logit stats, norm trajectory, anchor descriptors).
6. **Gated Fusion:** Sigmoid gate does element-wise re-weighting of concatenated [sequential; global] vector.
7. **Classifier Head:** Three-layer MLP -> hallucination logit.
8. **Contrastive Head:** Two-layer projection head for supervised contrastive learning.

#### Inputs
- **Sequential:** Per-layer descriptors S in R^{K x d_s} where each descriptor combines:
  - Last-token representation h^(l)
  - Sequence mean h_bar^(l)
  - Distributional statistics: L2 norm, mean, std, min/max, sparsity, kurtosis, MAD.
- **Global:** Top-k next-token probabilities, entropy, logit std, norm trajectory stats, anchor layer descriptors, cross-feature interactions.

#### Output Classes
- **Binary:** Hallucination logit (AUROC reported).

#### Training Data & Labeling
- HaluEval, TriviaQA (synthetic hard negatives).
- Composite loss: BCE + focal + asymmetric + contrastive + label smoothing.
- Data augmentation: Mixup, CutMix.
- 5-fold stratified CV with out-of-fold stacking (6 classifiers: RF, XGBoost, GradientBoosting, LightGBM, LogReg, SVM -> logistic meta-regressor).

#### Reported Performance
- **AUROC:** 98.43% (HaluEval, Mistral-7B) | 98.55% (HaluEval, LLaMA2-7B) | 98.30% (TriviaQA, Mistral-7B).
- Beats Neural CDEs (95.4%), SAPLMA (89.4%), MIND (94.5%).

#### Key Limitations
1. **Heavyweight:** A 6-layer transformer encoder + ensemble stacking is not a lightweight probe. It is orders of magnitude larger and slower than a 2-layer MLP.
2. **Post-generation:** It requires the full generation trajectory and global statistics (top-k probabilities, logit entropy) that are only available after the sequence is complete. Not a per-token generation-time detector.
3. **Binary:** Despite the rich architecture, the final output is still a single binary hallucination score.
4. **Complexity:** The out-of-fold stacking and multi-scale attention make it difficult to deploy in latency-sensitive applications.


---

## 7. Head-to-Head Architectural Comparison

| Method | Year | Probe Type | # Layers | Hidden Dims | Activation | Input | Token Granularity | # Output Classes | Trainable? |
|--------|------|------------|----------|-------------|------------|-------|-------------------|------------------|------------|
| **SAPLMA** | 2023 | MLP | 3 | [256->128->64] or similar | ReLU + Sigmoid | Single hidden state (selected layer) | Final prompt token | 2 (binary) | Yes |
| **SEPs** | 2024 | Linear probe | 1 | — | Linear/Logistic | Single hidden state (selected layer) | Last generated token | 2 (high/low SE) | Yes |
| **Single-Pass HalluDet** | 2025* | MLP | 3 | Unverified | ReLU (inferred) | Prefill hidden state | Last prefill token | 2 (binary) | Yes |
| **DCRD** | 2026 | Single-layer MLP | 1 | — | Unspecified | Attention-map fidelity features | Cross-attention weights | 2 (conflict/no) | Yes |
| **COCO** | 2026 | Neuron scorer | 0 | — | Contrastive loss | Per-neuron activation responses | All attention-head neurons | Neuron ranking | **No** |
| **ICR Probe** | 2025 | MLP | 4 | L->128->64->32->1 | Leaky ReLU + Sigmoid | Pooled ICR trajectory vector | Token-pooled across layers | 2 (binary) | Yes |
| **MultiHaluDet** | 2026 | Transformer + MLP | 6+3 | d=384, heads=8 | GELU, tanh, sigmoid | Multi-layer trajectory + global stats | Full sequence | 2 (binary) | Yes |
| **Approach B (Ours)** | — | MLP | 2 | 4096->2048->4 | ReLU + LayerNorm | Single generation-time hidden state | **Per-token** | **4** | Yes |

*Verification limited; see Section 3 notes.


---

## 8. Novelty Verdict: Will Reviewers Dismiss Approach B as "Just Another SAPLMA Variant"?

### 8.1 The Brutal Honest Assessment

**Yes — if the paper is written poorly.** A 2-layer MLP on a single hidden state is, architecturally, the simplest entry in the entire table above. By 2025-2026, the field has progressed to:

- **Deeper MLPs:** SAPLMA (3-layer), ICR Probe (4-layer).
- **Trajectory models:** ICR Probe (pooled cross-layer dynamics), MultiHaluDet (6-layer transformer over layer sequences).
- **Alternative features:** DCRD (attention maps), COCO (neuron-level causality), SEPs (entropy proxy).
- **Training-free methods:** COCO requires no probe training at all.

A reviewer looking *only* at the probe architecture will see a **regression** relative to SOTA, not an advance.

### 8.2 Where the Novelty Actually Lies (If Defended Correctly)

Approach B is **not** novel because of its layer count or hidden dimension. It is potentially novel because of **what it does with the simplest possible architecture** and **when it does it**:

| Differentiator | Why It Matters | Competitor Gap |
|----------------|----------------|----------------|
| **4-way classification** | Enables nuanced interventions (e.g., "regenerate" for hallucination, "request clarification" for contradiction, "confidence warning" for uncertainty). | *All* competitors above output binary scores. |
| **Generation-time, per-token** | Detects emerging problems while the model is writing, not after the fact. | SAPLMA, SEPs, Single-Pass HalluDet, MultiHaluDet operate post-hoc or prefill-only. |
| **Heuristic synthetic labeling** | Training data is generated without human annotation by simulating the 4 cognitive states. | Most methods rely on expensive correctness annotations or multi-sample SE computation. |
| **Neuroscience grounding (ACC)** | The 4 classes map to Anterior Cingulate Cortex conflict-monitoring theory, providing a theoretical framing. | Competitors are engineering papers without cognitive grounding. |

### 8.3 The Minimum Bar for Reviewer Acceptance

To avoid dismissal, the paper must empirically demonstrate **at least one** of the following:

1. **The 4-way taxonomy improves intervention utility.**
   - Show that a binary probe (trained on the same data, same architecture, just 1 output) leads to worse downstream task outcomes because it cannot distinguish "uncertain but correct" from "confidently hallucinated."

2. **Generation-time extraction beats prompt-time extraction.**
   - Ablate: train an identical 2-layer MLP on (a) prefill states vs. (b) generation-time states. Show that (b) achieves higher AUROC on a token-level hallucination benchmark.

3. **Architectural simplicity is a feature, not a bug.**
   - Show that the 2-layer MLP matches or exceeds the 4-layer ICR Probe / 3-layer SAPLMA on your data, proving that depth is unnecessary when the labeling taxonomy is richer. Or, show that MultiHaluDet is 50x slower with only marginal gain on your target domain.

4. **The heuristic labels are high-quality.**
   - Human evaluation: ask annotators to rate the correctness of the 4-class labels on a held-out set. If agreement is high, the synthetic pipeline is a valid contribution in its own right.


### 8.4 Risk Factors

| Risk | Severity | Mitigation |
|------|----------|------------|
| **"Just a small MLP"** | High | Emphasize the integration (entropy + consistency + detector) and the 4-way taxonomy in the abstract/intro. Do not lead with architecture details. |
| **"SAPLMA already did hidden states"** | High | Explicitly contrast prompt-time vs. generation-time in Table 1. Cite SAPLMA as precedent, then show the temporal shift. |
| **"DCRD also does conflict classification"** | Medium | Clarify that DCRD is attention-based, binary, and RAG-specific. Approach B is hidden-state-based, 4-class, and general-domain. |
| **"Single-Pass HalluDet is faster and deeper"** | Medium | Benchmark latency head-to-head. If HalluDet is real and achieves 0.89 AUC with 3 layers, you must match or exceed it, or argue that 4-way output justifies the trade-off. |
| **"MultiHaluDet crushes you on AUROC"** | Medium | Acknowledge that MultiHaluDet is a post-hoc heavyweight method. Position Approach B as the real-time, edge-deployable alternative. |
| **COCO is training-free** | Low | COCO is not a runtime detector. It is an analysis tool. No direct competition. |

### 8.5 Final Verdict

> **Approach B, in isolation, is not architecturally novel.** A 2-layer MLP classifying a single hidden state into 4 categories will not impress reviewers on its engineering merits alone.
>
> **However, the combination of (generation-time x per-token x 4-way taxonomy x neuroscience framing x synthetic heuristic labeling) is not found in any single competitor.** If the paper frames the contribution as a *coordinated inference-time verification layer* rather than "we built a better MLP probe," and if the experiments rigorously show that the 4-way classification enables interventions that binary probes cannot, the approach is defensible.
>
> **Recommendation:** Do **not** submit Approach B as a standalone architecture paper. Position it as one module of an integrated system (entropy monitor + consistency checker + latent conflict detector). The novelty is in the coordination and the cognitive taxonomy, not in the MLP depth.


---

## 9. References (Key Sources)

1. Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* arXiv:2304.13734.
2. Kossen, J., et al. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* ICML Workshop / arXiv:2406.15927.
3. Zhang, J., et al. (2025). *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs.* arXiv:2507.16488.
4. Zhang, J., et al. (2026). *Multilingual Hallucination Detection via LLM Hidden State Probing (MultiHaluDet).* arXiv:2605.24919.
5. Zhang, J., et al. (2026). *Modeling Implicit Conflict Monitoring Mechanisms against Stereotypes in LLMs (COCO).* arXiv:2605.09647.
6. Li, J., et al. (2026). *Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding (DCRD).* arXiv:2605.12185.
7. Farquhar, S., et al. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature, 630(8017), 625-630.
8. Chen, L., et al. (2025). *Uncertainty Quantification for Hallucination Detection in LLMs: Foundations, Methodology, and Future Directions.* arXiv:2510.12040.
9. Dasgupta, I., et al. (2025). *HalluShift: ...* (single-pass inter-layer consistency).
10. Anonymous. (2026). *HALT* (single-pass hidden-state hallucination detection).

---

*Review compiled by sub-agent. Independent verification attempted for all sources; Single-Pass HalluDet details rely on project-internal competitive tracking where external retrieval was incomplete.*
