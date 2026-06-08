# Honest Assessment: AAAI Publishability

## Current State

### What We Have
- Entropy monitor (well-known, not novel)
- Simple MLP detector trained on HaluEval prediction-error features
- Layer-pair prediction errors + leaky temporal integration
- 4-way taxonomy (factual/uncertain/hallucinated/contradictory)
- Small-scale evaluation: 12 samples showing +16.7% accuracy

### What We Don't Have
- No comparison against DoLa, ITI, SAPLMA, SEP, SelfCheckGPT
- No proper intervention mechanism (only warns, doesn't correct)
- No ablations showing each component matters
- Evaluation on one small model (Qwen2.5-1.5B)
- No statistical significance
- Training on 1K of 10K available examples

---

## Verdict: Not Currently AAAI-Ready

**Main conference AAAI requires:**
1. Strong technical novelty OR breakthrough empirical results
2. Comprehensive experiments on standard benchmarks
3. Comparison with strong baselines
4. Rigorous ablations and analysis
5. Theoretical or principled justification

We are currently at **prototype stage**, not **publication stage**.

---

## What's Genuinely Novel (Potential)

1. **Predictive coding framing for LLM hallucination detection**
   - Layer-pair prediction errors as a signal
   - Hierarchical rather than flat
   - This is conceptually distinct from DoLa (logits contrast) and SAPLMA (raw hidden states)

2. **Leaky temporal integration**
   - Most probes are per-step; ours accumulates over sequence
   - Could capture drift patterns that single-step methods miss

3. **4-way taxonomy**
   - hallucination / contradiction / uncertainty / factual
   - Literature is mostly binary
   - But we haven't really validated it empirically

---

## What's NOT Novel (Hurts AAAI Chances)

1. **Core detector is SAPLMA with engineered features**
   - MLP on hidden states → Azaria & Mitchell 2023
   - Prediction errors are just |h_l1 - h_l2|
   - Not a fundamentally different learning paradigm

2. **HaluEval dataset is standard**
   - Nothing new in the training data

3. **Entropy monitoring is 3+ years old**
   - Malinin & Gales 2021, Kuhn et al. 2023, etc.

4. **No actual intervention at generation time**
   - We detect but don't effectively correct
   - Logits biasing / activation editing not implemented

---

## Gap Analysis: What Would Make This AAAI-Worthy

### Option A: Stronger Technical Innovation

Make the predictive coding connection rigorous:
- Formalize prediction errors as free energy minimization
- Derive an update rule that modifies hidden states to reduce prediction error
- Show this is equivalent to (or better than) known methods in some regime
- Prove something: convergence bounds, error reduction guarantees

**Target contribution:** "A Neuro-Inspired Inference-Time Algorithm for Hallucination Reduction via Hierarchical Prediction Error Minimization"

### Option B: Much Stronger Empirics

Keep the method similar but scale experiments massively:
- 5+ models: Llama-3-8B, Mistral-7B, Qwen2.5-7B, Phi-4, Gemma-2-9B
- 5+ benchmarks: HaluEval, TruthfulQA, TriviaQA, FACTOR, SelfCheckGPT
- 8+ baselines: SAPLMA, DoLa, ITI, SEP, SelfCheckGPT, EigenScore, MHAD, entropy
- Metrics: AUC-ROC, accuracy, F1, hallucination rate, token overhead
- Ablations: layer hierarchy, temporal integration, prediction errors vs raw states
- Statistical tests: paired t-tests, confidence intervals

**Target contribution:** "Comprehensive study showing prediction-error probes outperform existing single-pass detectors"

### Option C: Novel Intervention Mechanism

Actually change generation based on detection:
- When conflict detected: bias logits toward factual tokens
- Or: shift activations along "truthful" direction (like ITI but guided by prediction error)
- Or: introduce abstention / "I don't know" token generation
- Show that intervention improves final output quality

**Target contribution:** "Self-Correcting LLMs via Real-Time Prediction Error Feedback"

---

## Realistic Publication Path

### 2-3 months of focused work → Workshop paper or specialized conference
- Scale to full HaluEval
- Add 2-3 baselines (SAPLMA, DoLa, entropy)
- Run on one 7B model
- Add proper intervention (logits biasing)
- Target: EMNLP Findings, COLING, EACL, or AAAI workshop

### 6+ months of focused work → Main conference possibility
- Implement Option A or C above
- Run comprehensive experiments (Option B)
- Theoretical justification
- Target: AAAI, ICLR, ACL main conference

---

## Recommendation

**Do not target AAAI 2027 with the current approach.** The framing is promising but the empirical and technical depth is insufficient.

**Better strategy:**
1. Decide if we want to pursue **theory** (Option A) or **systems** (Option B/C)
2. If theory: formalize the predictive coding connection
3. If systems: build a proper intervention mechanism and run comprehensive benchmarks
4. Target a workshop or specialized venue first, then iterate toward AAAI

The neuroscience angle is a **differentiator**, but it needs to be more than marketing — it needs to produce a genuinely better algorithm or provably useful insight.
