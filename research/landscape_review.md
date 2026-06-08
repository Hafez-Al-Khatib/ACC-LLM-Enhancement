# Hallucination Detection Landscape Review

## Research Date: 2026-05-28
## Focus: Inference-time, neuro-inspired, token-efficient methods

---

## 1. Method Taxonomy

### A. Multi-Pass / Sampling-Based (High Cost, Post-Hoc)
| Method | Mechanism | Cost | When |
|---|---|---|---|
| **SelfCheckGPT** (Manakul et al., 2023) | Generate N answers, measure consistency | N× forward passes | After generation |
| **Semantic Entropy** (Kuhn et al., 2023) | Sample N times, cluster meaning | N× forward passes | After generation |
| **Self-Consistency** | Majority vote across samples | N× forward passes | After generation |

**Verdict:** Strong accuracy but violate our constraint — they require multiple generations.

### B. Single-Pass Uncertainty (Training-Free, Cheap)
| Method | Mechanism | Accuracy | Limitation |
|---|---|---|---|
| **Token Entropy** | H(pθ(·\|y&lt;t)) per step | Moderate | Models can be confidently wrong |
| **Window-Entropy** | max over w-token window | Moderate | Same calibration issues |
| **Perplexity** | exp(-avg log-prob) | Low | Insensitive to local errors |
| **LN-Entropy** | Log-normalized entropy | Moderate | Brittle to prompt shifts |

**Verdict:** Our entropy monitor falls here. Fast but needs calibration. Research confirms models hallucinate with high confidence (Simhi et al., 2025), so entropy alone is insufficient.

### C. Hidden-State Probes (Supervised, Single-Pass)
| Method | Architecture | Training Data | AUC (HaluEval) |
|---|---|---|---|
| **SAPLMA** (Azaria & Mitchell, 2023) | MLP on hidden states | True/False statements | 79% |
| **SEP** (Kossen et al., 2025) | Linear probe → semantic entropy | SE-derived targets | ~75% |
| **MHAD** (Zhang et al., 2025) | MLP on multi-head attention | HaluEval | 64% |
| **EigenScore** (Chen et al., 2024) | Log-det of hidden-state covariance | Training-free | 61% |
| **HaloScope** (Du et al., 2024) | SAE on hidden states | Unlabeled gen | 84% |
| **HalluSAE** (2026) | Sparse autoencoder | HaluEval | **93%** |

**Verdict:** This is where our detector competes. SAPLMA uses a simple MLP on last-layer hidden states and gets 79% AUC. Our PredictiveCodingDetector is more complex but was trained on garbage data. Key insight: **simpler architecture + better data beats complex architecture + bad data**.

### D. Representation Editing (Inference-Time Intervention)
| Method | Mechanism | Modifies | Cost |
|---|---|---|---|
| **ITI** (Li et al., 2023) | Shift activations along truth directions | Hidden states | Single pass |
| **TruthX** (Zhang et al., 2024) | Autoencoder: factual vs semantic space | Hidden states | Single pass |
| **TruthForest** (Chen et al., 2024) | Orthogonal probes + activation edit | Hidden states | Single pass |
| **DoLa** (Chuang et al., 2023) | Contrast logits: early vs late layers | Logits | Single pass |
| **ICD** (Zhang et al., 2023) | Contrast original vs amateur model | Logits | Single pass |
| **PLI** (2025) | Interpolate premature layers | Layer params | Single pass |

**Verdict:** These are our closest competitors. They modify generation in real-time without extra tokens. However:
- ITI needs pre-computed truth directions per model
- DoLa only uses logits contrast, not hidden-state geometry
- None use temporal dynamics or hierarchical prediction errors

### E. Neuroscience-Inspired (Our Niche)
| Concept | Source | Application to LLMs |
|---|---|---|
| **Predictive Coding** | Rao & Ballard (1999), Friston (2005) | Almost none directly |
| **Free Energy Principle** | Friston (2010) | Rarely applied to transformers |
| **ACC Conflict Monitoring** | Botvinick et al. (2001) | Not applied to LLMs |
| **Hierarchical Prediction Errors** | Friston (2008) | DoLa uses layer contrast (loosely related) |

**Verdict:** Our approach is genuinely novel in applying predictive coding + ACC conflict monitoring to LLM inference. No existing method explicitly frames detection as hierarchical prediction error minimization.

---

## 2. Key Datasets for Training Detectors

| Dataset | Size | Labels | Best For |
|---|---|---|---|
| **HaluEval** | 30K (QA/Dial/Sum) | Hallucination/Factual | Training detectors |
| **TruthfulQA** | 817 questions | Best/correct/incorrect answers | Evaluating abstention |
| **TriviaQA** | 650K | Verified answers | Contrastive training |
| **SelfCheckGPT (WikiBio)** | ~500 | Accurate/Major-inaccurate/Minor | Consistency eval |
| **True-False (Azaria)** | 6,084 | True/False per topic | SAPLMA training |
| **FactCHD** | 58K | Factual conflict | Factual reasoning |
| **FaithDial** | 22K | BEGIN labels (faithful/hallucinated) | Dialogue hallucination |

**Our mistake:** We trained on 10 prompts with self-consistency pseudo-labels. Literature uses thousands of labeled examples (HaluEval, True-False). **We need real labeled data.**

---

## 3. Critical Research Findings

### Finding 1: Single uncertainty signals are insufficient
> "LLMs show the tendency to remain highly confident even when hallucinating" — Simhi et al., 2025

This means entropy-only (our current working mode) will miss "confident hallucinations." Need hidden-state signal as complement.

### Finding 2: Layer-contrast methods work (DoLa)
> "Factual information is encoded in distinct layers... contrast early vs late layers" — Chuang et al., 2023

DoLa improves factuality by 5-15% on TruthfulQA. This validates our layer-pair prediction error idea, but DoLa only uses logits, not hidden states.

### Finding 3: Temporal dynamics matter
> "Full residual-stream analysis is costly... we train on compact final-layer entropy summaries" — Ali et al., 2025

Our leaky integrator (temporal decay) captures sequence-level patterns that single-step probes miss. This is a genuine differentiator.

### Finding 4: Simpler probes can work well
SAPLMA: single MLP on last-layer hidden state → 79% AUC.
HalluSAE: sparse autoencoder → 93% AUC.

Our detector has 4M parameters. SAPLMA probably has <100K. **Complexity without data is a liability.**

### Finding 5: Interventions should not corrupt text
Inserting `[HALLUCINATION]` markers into generated text:
- Breaks downstream parsing (yes/no QA, structured output)
- Confuses the model on next-token prediction (markers become context)
- Makes evaluation impossible

**Better approaches from literature:**
- Logits biasing (DoLa, ICD)
- Activation shifting (ITI)
- Temperature adjustment for uncertain tokens
- Early stopping or abstention tokens

---

## 4. Where Our Approach Fits

### Genuine Novelties
1. **Predictive coding framing** — no existing LLM method uses this explicitly
2. **Hierarchical prediction errors between layer pairs** — deeper than DoLa's single contrast
3. **Leaky temporal integration** — captures dynamics that static probes miss
4. **4-way taxonomy** (factual/uncertain/hallucination/contradiction) — richer than binary probes
5. **ACC conflict monitoring** — neuro-inspired, not just statistical

### Weaknesses vs. State-of-the-Art
1. **Training data** — 10 prompts vs. HaluEval's 30K labeled examples
2. **Architecture complexity** — 4M params vs. SAPLMA's ~50K
3. **Intervention mechanism** — text markers vs. activation editing (ITI) or logits contrast (DoLa)
4. **Evaluation** — no comparison against DoLa, ITI, SAPLMA baselines
5. **Calibration** — fixed threshold vs. learned/empirical calibration

---

## 5. Recommended Pivot

Based on this research, I recommend we restructure our approach:

### Phase 1: Fix the Foundation (1-2 days)
1. **Download HaluEval** and train detector on real labels
2. **Simplify detector** — replace BiLSTM with MLP or 2-layer network (like SAPLMA)
3. **Remove text markers** — switch to logits biasing or temperature adjustment
4. **Add proper baselines** — implement DoLa, ITI, SAPLMA for comparison

### Phase 2: Validate Novelty (2-3 days)
1. **Show prediction errors > logits contrast** — compare our layer-pair errors vs. DoLa
2. **Show temporal integration helps** — ablate leaky integrator
3. **Show 4-way taxonomy is useful** — measure precision per class
4. **Benchmark on standard datasets** — HaluEval, TruthfulQA, TriviaQA

### Phase 3: Position Paper (ongoing)
Frame as: "Predictive Coding for Self-Aware LLM Inference"
- Novelty: hierarchical prediction errors + temporal dynamics + neuroscience framing
- Not just another probe — it's a **mechanism** for self-awareness
- Not just another decoder — it's **biologically grounded**

---

## 6. Immediate Action Items

1. **Download HaluEval from HuggingFace** — `levertco/HaluEval` or `pminervini/HaluEval`
2. **Implement SAPLMA baseline** — MLP on last-layer hidden states
3. **Implement DoLa baseline** — contrastive decoding between layers
4. **Rewrite detector** — simpler architecture, trained on HaluEval
5. **Design non-destructive intervention** — logits biasing instead of markers
