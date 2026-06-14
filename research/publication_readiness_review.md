# Honest Publication Readiness Review

**Project:** ACC — Neuroscience-Inspired Hallucination Detection  
**Review Date:** June 2026  
**Reviewer:** Kimi Code CLI (technical audit + empirical assessment)

---

## Executive Verdict

**Current state is not publishable as a novel research paper.**

It is a **promising prototype** with a coherent conceptual framing, but it lacks the empirical depth, theoretical rigor, and systematic comparison required for publication in a top-tier venue (AAAI, NeurIPS, ACL, ICLR, EMNLP). With substantial additional work — primarily large-scale experiments and stronger baselines — it could become a workshop paper or a mid-tier conference submission. A top-tier archival paper would require a major expansion.

---

## 1. What Is Working

### 1.1 Conceptual Framing

The neuroscience-inspired angle (predictive coding + anterior cingulate conflict monitoring + top-down cognitive control) is **genuinely interesting** and distinguishes the work from purely engineering-driven hallucination detection papers. The mapping is:

- Hierarchical prediction errors → neural prediction-error signals
- Leaky integrator → temporal evidence accumulation
- Conflict score → ACC activity
- Regeneration with uncertainty priming → prefrontal control

This is the paper's strongest asset. If developed formally, it could be a real contribution.

### 1.2 Engineering Pipeline

The codebase now has:
- A working detector architecture (`PredictiveCodingDetector`)
- A training pipeline with focal loss and soft labels
- An intervention engine (`ACCInterventionEngine`)
- Baseline implementations (DoLa, SAPLMA, Entropy)
- A unified evaluation script
- A Colab notebook and RTX 4090 scripts
- A code audit that fixed several critical bugs

This is **good infrastructure** for a research project.

### 1.3 Detector Training

The custom detector trained on model-specific data achieves **83.3% validation accuracy**. This shows the core learning signal is present — the model's hidden states do contain discriminable information about hallucination risk for that specific model.

---

## 2. Critical Weaknesses

### 2.1 Empirical Scale Is Far Too Small

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Evaluation samples | 10 (CPU) / 43 (4090 script, not yet completed) | Need 500–2,000 for confidence |
| Models tested | Qwen2.5-1.5B | Need at least 3–5 models (1.5B, 7B, 13B+) |
| Benchmarks | Ad-hoc prompts | Need HaluEval, TruthfulQA, PubMedQA, SelfCheckGPT test sets |
| Human/LLM-as-judge | Substring matching | Need reliable automated judge + human spot-check |
| Statistical testing | None | Need paired t-tests, bootstrap CIs, significance |

**A 10-sample evaluation is not an experiment. It is a smoke test.** Even 43 samples is a pilot. Publication-quality work needs hundreds to thousands of examples with confidence intervals.

### 2.2 Results Do Not Demonstrate Improvement

Current CPU results on Qwen2.5-1.5B:

| Method | Accuracy |
|--------|----------|
| Baseline | 50% |
| Entropy | 50% |
| DoLa | 50% |
| SAPLMA | 50% |
| ACC | 60% |

The 60% for ACC is driven entirely by factual questions (75% vs. 50% baseline). On hallucination and uncertain questions, **all methods are at 50% — indistinguishable from chance**.

A paper needs to show *reliable, statistically significant improvement* on the hard cases (hallucination detection), not just on easier factual recall.

### 2.3 Baselines Are Not Strong Enough

Implemented baselines:
- Entropy threshold
- DoLa (simplified post-hoc version)
- SAPLMA (tiny inline-trained MLP)

Missing baselines:
- **SelfCheckGPT** (sampling-based consistency)
- **LM vs. LM** (perplexity-based detection)
- **TruthfulQA-style probes**
- **DoLa proper** (layer-contrast during generation, not post-hoc)
- **LLM-as-judge** (e.g., GPT-4o as a detector baseline)
- **Entropy / log-probability calibrated per-model**

Reviewers will ask: "Why isn't SelfCheckGPT in the comparison?"

### 2.4 Intervention Is Weak

Current intervention: prepend "Wait, let me reconsider" and bump temperature.

This is a **very weak form of intervention**. It does not:
- Shift logits toward uncertainty tokens
- Use retrieval or grounding
- Constrain the decoding vocabulary
- Apply feedback to the detector

A paper claiming "hallucination prevention" needs a prevention mechanism that demonstrably reduces hallucination rate, not just re-rolls the dice.

### 2.5 Theoretical Contribution Is Underdeveloped

The neuroscience framing is described informally. There is:
- No formal predictive coding objective
- No derivation of why layer-pair prediction errors should correlate with hallucination
- No analysis linking the leaky integrator to evidence accumulation theory
- No hypothesis about *which* layers matter and why
- No connection to cognitive control models from neuroscience

Without formalization, the framing risks being dismissed as "inspirationware."

### 2.6 Judge Function Is Unreliable

Current judging uses substring matching (e.g., "did not", "uncertain", "as an ai"). This is:
- Brittle ("not" matches "nothing")
- Unable to handle paraphrases
- Not validated against human judgments

A paper needs a validated evaluation protocol.

### 2.7 No Ablation Studies

We do not know *which* components matter:
- Are hierarchical prediction errors better than last-layer features?
- Does temporal integration help?
- Does the 3-way classification help over binary?
- Which layer pairs matter?
- Does per-prompt calibration help?

Ablations are mandatory for any architecture paper.

---

## 3. What Would Be Needed for Publication

### 3.1 Minimum Viable Workshop Paper

To submit to a workshop (e.g., NeurIPS/ICML workshop on hallucination, or a NLP workshop):

1. **Run the 4090 evaluation** on 7B with 200+ samples and report results.
2. **Add 2–3 strong baselines** (SelfCheckGPT, proper DoLa, LLM-as-judge).
3. **Add LLM-as-judge** for labels instead of substring matching.
4. **Add statistical tests** (paired t-test, bootstrap CIs).
5. **Add ablations** (remove temporal integration, remove hierarchical errors, binary vs. 4-way).
6. **Write a clear related work section** positioning against DoLa, SAPLMA, SelfCheckGPT.

**Estimated effort:** 3–4 weeks of focused experimentation.

### 3.2 Strong Conference Paper (AAAI / ACL / EMNLP)

For a top-tier archival paper:

1. **Large-scale benchmarks:** HaluEval (full 10K or 1K subset), TruthfulQA, PubMedQA, SVAMP or a custom 1K+ dataset.
2. **Multiple models:** At least 4 models spanning sizes (1.5B, 7B, 13B, 70B if possible).
3. **Stronger intervention:** Implement logit-shifting or constrained decoding with grounding.
4. **Theoretical grounding:** Formalize the predictive coding objective; derive the conflict score; connect to existing work on surprise/conflict in neural networks.
5. **Human evaluation:** 100–200 examples judged by humans, with inter-annotator agreement.
6. **Extensive ablations and analysis:** Per-layer analysis, per-category breakdown, failure modes, qualitative examples.
7. **Reproducibility:** Full code, configs, seeds, prompts, model checkpoints.

**Estimated effort:** 3–6 months full-time research.

### 3.3 What Could Make This Paper Special

The neuroscience angle is the differentiator. To make it publication-worthy:

- **Formalize predictive coding in LLMs:** Define a layer-wise prediction error loss and show it correlates with hallucination probability across layers.
- **Interpretability contribution:** Show that the detector's conflict scores align with human uncertainty judgments or with factual vs. counterfactual prompts.
- **Intervention contribution:** Show that the ACC signal can be used to *reduce* hallucination rate measurably (not just detect it).
- **Novel taxonomy:** The 4-way classification (supported/unsupported/uncertain/contradictory) could be a contribution if validated empirically.

---

## 4. Honest Comparison to Related Work

| Method | What it does | Our current edge | Our current deficit |
|--------|--------------|------------------|---------------------|
| **DoLa** | Contrast early/late layers during decoding | We add temporal integration + explicit conflict classes | DoLa is simpler, already published, and our post-hoc version is weaker |
| **SAPLMA** | MLP on last-layer hidden states | We use hierarchical prediction errors across layers | SAPLMA has stronger published results and standard benchmarks |
| **SelfCheckGPT** | Consistency across multiple samples | We run during generation, not post-hoc | SelfCheckGPT is much stronger empirically and has human evaluations |
| **ITI / SEP** | Shift activations toward truthful directions | We detect conflict explicitly | They have stronger benchmark results and theoretical grounding |

**Current status:** We are not yet beating or clearly differentiating from these methods.

---

## 5. Realistic Assessment by Venue

| Venue Type | Chance of Acceptance | Why |
|------------|---------------------|-----|
| AAAI / NeurIPS / ICML main | <5% | No large-scale results, weak theory, unproven improvement |
| ACL / EMNLP main | <10% | No standard NLP benchmarks, weak baselines, small scale |
| Top workshop | 20–30% | If 7B results are strong and baselines are added |
| ArXiv preprint | 100% | But would not be taken seriously by reviewers |
| Reputable mid-tier conference | 15–25% | Would need at least 2x the current empirical work |

---

## 6. Recommended Immediate Priorities

If the goal is publication, the next 2 weeks should focus exclusively on:

1. **Run the 4090 evaluation on Qwen2.5-7B** with the existing 43-sample set.
2. **Expand to 200+ samples** using HaluEval + TruthfulQA + custom prompts.
3. **Implement LLM-as-judge** (e.g., via API or local judge model) for reliable labels.
4. **Add SelfCheckGPT baseline** — this is the most important missing baseline.
5. **Add statistical testing** to all comparisons.

Only after these are done should we invest in theory or additional architecture.

---

## 7. Conclusion

The project has **strong conceptual foundations** and **solid engineering**, but the empirical story is currently **too thin for publication**. The critical missing pieces are:

1. Scale (hundreds to thousands of examples)
2. Strong baselines (especially SelfCheckGPT)
3. Reliable evaluation (LLM-as-judge + statistical testing)
4. A stronger intervention mechanism
5. Formal theoretical grounding

**Honest recommendation:** Do not submit in the current state. Treat the next 3–4 weeks as a focused effort to produce a workshop-ready paper, and the next 3–6 months for a top-tier conference paper if results justify it.

The neuroscience-inspired framing is worth pursuing, but it must be backed by numbers that prove the idea works.
