# AAAI Readiness Audit — Honest and Harsh

**Project:** ACC — Neuroscience-Inspired Hallucination Detection  
**Audit Date:** June 2026  
**Auditor:** Kimi Code CLI  
**Scope:** Code, experiments, theory, and presentation readiness for AAAI submission.

---

## Executive Verdict

**This project is not ready for AAAI submission.**

It is a well-engineered prototype with an interesting conceptual angle, but it lacks the empirical volume, rigorous baselines, proven novelty, and polished presentation required for a top-tier AI conference. In its current state, submission to AAAI would likely be rejected in the first round of reviews.

The honest path forward is:
- **Workshop paper:** 4–6 weeks of focused experiments if 7B results are strong.
- **AAAI / ACL / EMNLP main:** 4–6 months of full-time research, assuming results justify the effort.

---

## 1. What AAAI Actually Requires

AAAI reviewers evaluate papers on four axes:

1. **Novelty and Significance:** Is the idea new? Does it matter?
2. **Technical Correctness and Rigor:** Is the method sound? Are claims justified?
3. **Empirical Depth:** Are experiments large-scale, fair, and statistically sound?
4. **Clarity and Reproducibility:** Can others understand and replicate the work?

This audit evaluates the project against each axis. Gaps are rated **CRITICAL**, **HIGH**, **MEDIUM**, or **LOW**.

---

## 2. Novelty and Significance

### 2.1 Claimed Contribution

The project proposes a neuroscience-inspired approach to hallucination detection:
- Hierarchical predictive coding in transformer hidden states.
- Anterior cingulate cortex (ACC) conflict monitoring.
- Temporal integration via a leaky integrator.
- Generation-time intervention triggered by conflict scores.

### 2.2 Harsh Assessment

**The conceptual framing is interesting but not yet a proven contribution.**

- **Predictive coding in LLMs is not new.** DoLa (Chuang et al., ICLR 2024) already contrasts layer-wise predictions. SAPLMA (Azaria & Mitchell, 2023) already uses hidden states to detect lies. The added value of "ACC conflict monitoring" is currently **narrative, not empirical**.
- **The neuroscience mapping is analogical, not formal.** The project invokes Friston, Botvinick, and Rao & Ballard, but there is no theorem, no derived objective, and no empirical validation that the model's prediction errors behave like cortical prediction errors. Reviewers will call this "inspirationware."
- **The intervention is not novel enough.** Forcing an uncertainty phrase when a detector fires is a straightforward heuristic. It does not use retrieval, grounding, feedback control, or constrained decoding.

### 2.3 What Would Make It Novel for AAAI

To be novel, the paper needs at least one of:
1. **A formal predictive coding objective** for LLMs that is optimized or analyzed.
2. **Empirical evidence that ACC-style conflict signals outperform existing layer-contrast methods** (DoLa, logit lens, hidden-state classifiers) across benchmarks.
3. **A theoretically justified intervention** (e.g., logit shifting derived from a control objective, or a gating mechanism with provable properties).
4. **A new taxonomy or dataset** validated by human annotators.

Currently, none of these are present.

**Gap Severity: HIGH**

---

## 3. Technical Correctness and Rigor

### 3.1 What's Sound

- The `PredictiveCodingDetector` architecture is internally consistent.
- The leaky integrator is implemented correctly.
- 99 unit tests pass, covering basic generation, detection, and integration paths.
- Several critical bugs from the earlier audit have been fixed:
  - Synthetic logits inversion in `HaluEvalDetector`.
  - Post-hoc hidden-state mismatch in `acc_integration.py`.
  - Gradient accumulation in `get_layer_contributions`.

### 3.2 What's Still Broken or Questionable

#### CRITICAL: README Does Not Match the Project

The `README.md` describes a **Mistral 7B fine-tuning project with QLoRA** across Medical, Legal, Financial, STEM, and General verticals. The actual project is a **hallucination detection and intervention framework for Qwen2.5**. A reviewer reading the repo would be confused about what the paper is even about. This alone would sink a submission.

#### CRITICAL: No Paper Draft

There is no LaTeX/paper draft, no abstract, no introduction, no related work section, no experimental setup, no results section. Research notes and slides are not a paper.

#### HIGH: Detector Training Is Model-Specific and Tiny

- The "custom detector" is trained on **600 token-level examples** from Qwen2.5-1.5B.
- It achieves 83.3% validation accuracy, but:
  - The validation set is from the same model and distribution.
  - There is no cross-model generalization test.
  - There is no cross-domain generalization test.
- For AAAI, a detector trained on 600 examples for one model is a **pilot experiment**, not a validated method.

#### HIGH: HaluEvalDetector Is a Hack

`HaluEvalDetector` wraps a simple MLP and manually synthesizes 3-way / 2-way logits from a single scalar probability. The mapping:

```python
primary_logits[:, 0] = torch.log(1 - prob + 1e-6)
primary_logits[:, 1] = torch.log(prob + 1e-6)
primary_logits[:, 2] = -torch.abs(prob - 0.5) * 4
```

This is mathematically **ad hoc**. It forces a 3-way classification out of a binary detector. The "uncertain" class peaks at prob=0.5 but is not learned from data. A reviewer would ask: why not train a real 3-way classifier? Why not calibrate these logits?

#### HIGH: `acc_integration.py` Logic Is Fragile

- `_ACCLogitsProcessor` skips `gen_step == 0` for conflict detection, missing the first generated token.
- Regeneration multiplies logits by a multiplier, making the distribution **sharper** (lower effective temperature), which is the opposite of increasing diversity.
- The `confidence_score` is replicated across all batch items and derived from a globally interleaved sliding window.
- Text marker insertion splices at token boundaries and can corrupt Unicode / subword boundaries.

#### MEDIUM: Baselines Are Toy Implementations

- **DoLa:** Uses a post-hoc Jensen-Shannon divergence, not the real contrastive decoding from the paper. No proper premature/mature layer selection. No Matura or Matura+.
- **SAPLMA:** Trained inline on **4 examples** (2 factual, 2 hallucinated). This is not a serious baseline.
- **SelfCheckGPT:** Implemented as n-gram overlap across 5 samples. The real SelfCheckGPT uses sentence-level entailment, question generation, or LLM prompting. The current implementation is a caricature.
- **Entropy:** Standard threshold; fine as a weak baseline, but not enough.

#### MEDIUM: Device and Memory Handling

- Manual generation loops in `acc_intervention.py` and `baselines.py` do not use KV-cache acceleration. For 7B models and 500-sample benchmarks, this will be **~10× slower** than necessary.
- Hooks move hidden states to CPU on every forward pass, blocking the GPU.
- The code supports XPU/CUDA/CPU but has not been stress-tested at scale.

**Gap Severity: HIGH**

---

## 4. Empirical Depth

### 4.1 Current Results

All experiments are on **Qwen2.5-1.5B** on Intel Arc XPU:

| Experiment | Samples | Notes |
|------------|---------|-------|
| Ablation study | 10 hand-written prompts | Phrase injection reaches 90–100% |
| Benchmark eval (no LLM judge) | 10 HaluEval | ACC 10%, SAPLMA 90% (suspicious) |
| Benchmark eval (LLM judge) | 9 mixed | ACC 11.1%, all else 0% |
| 4090 eval | 0 completed | Script ready but not run |

### 4.2 Harsh Assessment

**These are not experiments. They are smoke tests.**

For AAAI, the expected empirical scale is:

| Requirement | Current | AAAI Minimum | Gap |
|-------------|---------|--------------|-----|
| Evaluation samples | 10–43 | 500–2,000 | **50–200×** |
| Models evaluated | 1 (1.5B) | 3–5 (1.5B, 7B, 13B+) | **3–5×** |
| Benchmarks | Ad-hoc + small HaluEval | HaluEval, TruthfulQA, PubMedQA, custom | **3–4×** |
| Human/LLM judge | Substring + tiny LLM judge | GPT-4o / Claude / human spot-check | **Unreliable** |
| Statistical tests | Bootstrap + t-test added | Still need power analysis | Partial |
| Error analysis | None | Per-category, per-layer, qualitative | **Missing** |

### 4.3 Specific Empirical Problems

#### CRITICAL: No 7B Results

The entire conceptual argument depends on the hypothesis that **larger models can use the ACC signal better**. This hypothesis is **untested**. The 4090 script exists but has not been run. Without 7B results, there is no empirical story.

#### CRITICAL: Judge Is Not Trustworthy

- Fallback judging uses substring matching (e.g., "not" matches "nothing").
- LLM-as-judge uses the **same 1.5B model** as the generator. It is too weak to judge reliably and often ignores instructions.
- There is no human evaluation or inter-annotator agreement.

#### HIGH: Results Are Contradictory and Unstable

- SAPLMA scores 90% on 10 HaluEval samples but 0% with LLM judge on 9 mixed samples. This instability suggests **overfitting to the tiny training set** and judge sensitivity.
- ACC scores 0–11% on benchmark tasks but 90–100% on hand-written ablation prompts. This suggests the **ablation prompts are too easy** and do not represent real hallucination detection.

#### HIGH: No Cross-Model or Cross-Domain Validation

A method that only works on Qwen2.5-1.5B with a detector trained on Qwen2.5-1.5B data is not a general contribution. AAAI reviewers will ask for results on at least Llama, Mistral, and Qwen families.

**Gap Severity: CRITICAL**

---

## 5. Clarity and Reproducibility

### 5.1 What's Good

- Code is well-structured into `src/`, `scripts/`, `configs/`, `tests/`.
- 99 tests pass.
- Git history is active and pushed to GitHub.
- There are setup scripts for XPU, CUDA, and 4090.
- Configs exist for different hardware targets.

### 5.2 What's Missing

#### CRITICAL: No Paper Draft

Again: there is no paper. AAAI submissions are 7 pages + references. The current deliverables are:
- `README.md` (about the wrong project)
- `research/publication_readiness_review.md`
- `research/neuroscience_framework.md`
- `results/slides/ACC_Weekly_Update_2026-06-28.pptx`

These are supporting materials, not a submission.

#### HIGH: Random Seeds Not Fully Controlled

- `torch.manual_seed(seed)` is called, but CUDA/XPU RNG state is not explicitly set in all paths.
- Different methods use different seeds and sampling strategies, making fair comparison difficult.

#### HIGH: Hyperparameters Are Scattered and Unexplained

- Thresholds (0.5, 1.5, 3.9, 0.1) are hardcoded without ablation or sensitivity analysis.
- Layer pairs are chosen without justification.
- Temperature and top-p differ across methods.

#### MEDIUM: No Reproducibility Checklist

No `REPRODUCIBILITY.md`, no exact command log, no pinned dependency versions for 4090, no container/conda env export.

**Gap Severity: HIGH**

---

## 6. Related Work and Positioning

### 6.1 Current State

The project mentions DoLa, SAPLMA, SelfCheckGPT, and predictive coding/cognitive control in the neuroscience framework doc. However:
- There is no formal related work section.
- There is no citation list in APA/BibTeX format ready for a paper.
- The comparison is superficial ("we add temporal integration").

### 6.2 Missing Related Work

For AAAI, the paper must position against:

1. **DoLa** (Chuang et al., ICLR 2024) — contrastive layer decoding.
2. **SAPLMA** (Azaria & Mitchell, 2023) — hidden-state lie detection.
3. **SelfCheckGPT** (Manakul et al., 2023) — consistency-based detection.
4. **ITI / SEP** (Li et al., 2023) — activation steering for truthfulness.
5. **LM vs. LM** (Cohen et al., 2023) — cross-model consistency.
6. **HaluEval** (Li et al., 2023) — benchmark dataset.
7. **TruthfulQA** (Lin et al., 2022) — benchmark dataset.
8. **Predictive coding in neural networks** (e.g., Lotter et al., 2016; Wen et al., 2020).
9. **Uncertainty quantification in LLMs** (e.g., Kuhn et al., 2023; Lin et al., 2023).

Without this, reviewers will say the paper lacks context.

**Gap Severity: HIGH**

---

## 7. Realistic Assessment by Venue

| Venue | Acceptance Probability | Why |
|-------|------------------------|-----|
| **AAAI main** | <5% | No large-scale results, weak theory, no paper draft. |
| **NeurIPS / ICML main** | <5% | Same issues; these venues are even more empirically demanding. |
| **ACL / EMNLP main** | <10% | No standard NLP benchmarks at scale, weak baselines. |
| **AAAI workshop** | 20–30% | If 7B results show clear improvement and baselines are strengthened. |
| **NeurIPS/ICML workshop on hallucination** | 25–35% | Good fit if experiments are scaled up. |
| **ArXiv preprint** | 100% | But would not be taken seriously without major revisions. |

---

## 8. Prioritized Roadmap to AAAI Readiness

### Phase 1: Minimum Viable Submission (6–8 weeks)

Focus exclusively on empirical credibility. Do not write theory yet.

1. **Run 4090 evaluation on Qwen2.5-7B** with 500+ samples across HaluEval, TruthfulQA, PubMedQA.
2. **Add proper LLM-as-judge** using a strong model (GPT-4o / Claude / Qwen2.5-72B) or human annotations for 200 examples.
3. **Strengthen baselines:**
   - Replace toy SAPLMA with a real detector trained on 1,000+ held-out examples.
   - Implement proper DoLa contrastive decoding.
   - Replace n-gram SelfCheckGPT with sentence-level consistency (e.g., via NLI or LLM).
   - Add a strong entropy/log-probability baseline calibrated per-model.
4. **Fix evaluation fairness:** same sampling strategy, same seeds, same max tokens for all methods.
5. **Add statistical testing** and report confidence intervals.
6. **Write the paper draft:** abstract, intro, related work, method, experiments, analysis, conclusion.
7. **Rewrite README** to match the actual project.

### Phase 2: Competitive AAAI Paper (3–4 months)

If Phase 1 results show ACC significantly outperforms baselines:

1. **Scale to multiple models:** Qwen2.5-1.5B, 7B, 14B; Mistral 7B; Llama 3.1 8B.
2. **Cross-domain evaluation:** factual QA, biography generation, medical QA, math word problems.
3. **Human evaluation:** 200+ examples with inter-annotator agreement.
4. **Formalize theory:** derive the predictive coding objective, justify layer-pair choices, connect leaky integrator to evidence accumulation.
5. **Stronger intervention:** implement retrieval-augmented abstention, constrained decoding, or feedback-controlled generation.
6. **Ablation studies:** remove temporal integration, remove hierarchical errors, binary vs. 4-way, per-layer analysis.
7. **Error analysis:** qualitative examples, failure modes, correlation with human judgments.
8. **Reproducibility package:** container, pinned deps, all seeds, all prompts, all checkpoints.

### Phase 3: Top-Tier Paper (4–6 months)

1. **Large-scale benchmarks:** full HaluEval (10K) or 1K subset, TruthfulQA, PubMedQA, custom 1K+ dataset.
2. **70B model results** if compute allows.
3. **Novel theoretical contribution:** prove or empirically validate a predictive coding principle in LLMs.
4. **New dataset or metric** validated by human annotators.
5. **Comparison with all SOTA methods** including API-based judges and detectors.

---

## 9. What Specifically Must Be True for AAAI

For this paper to have a real shot at AAAI, the following must be demonstrably true:

1. **ACC significantly improves over strong baselines** (not just entropy and toy SAPLMA) on 500+ examples with p < 0.05.
2. **The intervention reduces hallucination rate**, not just flags uncertain tokens.
3. **The neuroscience framing is more than analogy** — e.g., prediction errors correlate with hallucination probability across layers and models.
4. **Results generalize** across at least 3 model families.
5. **Evaluation is reliable** — LLM-as-judge or human evaluation with agreement metrics.
6. **Ablations show which components matter** — temporal integration, hierarchical errors, layer selection.
7. **A complete, well-written paper draft** exists with proper related work and reproducibility.

Currently, **0 of 7** are true.

---

## 10. Conclusion

This is a **promising research direction** with solid engineering, but it is **not an AAAI-ready paper**. The most honest recommendation is:

> **Do not submit to AAAI in the current state.**

Instead, treat the next 6–8 weeks as a sprint to produce:
1. Large-scale 7B results.
2. Strong baselines and reliable evaluation.
3. A complete paper draft.

If the 7B results show clear, statistically significant improvement over strong baselines, then a workshop submission is realistic. If the results are mixed, the project needs deeper theoretical and methodological work before targeting a top-tier venue.

The neuroscience-inspired angle is the right differentiator, but it must be backed by numbers, not just narrative.
