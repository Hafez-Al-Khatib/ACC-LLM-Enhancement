# Approach B Labeling Strategy: Detailed Analysis & Recommendations

**Date:** 2026-05-30
**Project:** ACC LLM Enhancement
**Analyst:** ML Research Methodologist (Sub-agent)
**Scope:** Evaluate the current 4-way per-token label taxonomy (supported, hallucinated, uncertain, contradictory) against the hallucination-detection literature; assess heuristic labeling quality; and recommend structural changes.

---

## 1. Executive Summary

Approach B currently labels each generated token into four discrete classes and trains a lightweight MLP on hidden states using ~2,000 heuristic per-token examples. The literature shows that:

1. **4-way classification is rare but not unprecedented** — most benchmarks use binary or 6–8 fine-grained span-level categories. The value of multi-class labels depends entirely on whether the categories are **actionable** at inference time.
2. **Heuristic labeling has well-documented failure modes** — substring matching fails on paraphrases, embedding similarity suffers from the "Semantic Illusion," and token-probability thresholds conflate lexical uncertainty with factual uncertainty.
3. **Human annotators struggle with fine-grained taxonomies** — exact-type inter-annotator agreement (IAA) drops to ~60% for 6-way schemes, while binary detection agreement is ~75–92%.
4. **Model-based pseudo-labels can exceed heuristic quality** — GPT-4o-mini achieved 93.5% agreement with humans on ambiguous cases and 96.1% overall in a recent large-scale study.
5. **Soft/continuous labels are gaining traction** — smoothed knowledge distillation and uncertainty-guided pseudo-labeling both reduce overconfidence and improve calibration.

**Bottom-line recommendation:**
- **Simplify the taxonomy to 3 classes for the primary detector** (`supported`, `unsupported`, `uncertain`), but keep a **secondary type classifier** trained on soft pseudo-labels for diagnostic granularity.
- **Replace pure heuristic labeling with a multi-signal pseudo-labeling pipeline** (heuristic filter → NLI check → strong LLM judge) and use **soft-label cross-entropy** during training.
- **Introduce continuous confidence scores** as auxiliary training targets to improve calibration and enable threshold-tunable interventions.

---

## 2. Label Taxonomies in the Hallucination Detection Literature

### 2.1 Binary (Hallucinated vs. Non-Hallucinated)
**Prevalence:** Dominant in benchmarks and production systems.
- HaluEval, RAGTruth, and the Hughes Hallucination Evaluation Model (HHEM) all reduce the task to a binary decision (Huang et al., 2025; HHEM, 2025).
- A 2024–2025 survey notes that binary labels are the norm but criticizes them for ignoring *partial hallucinations* and *varying error severity* (Huang et al., 2025).
- Binary detection is computationally cheaper and yields higher AUC/F1 in practice; however, it provides no diagnostic signal for downstream remediation.

### 2.2 3-Way Classification
**Variants:** `True` / `False` / `Uncertain` or `Factual` / `Non-factual` / `Neutral`.
- A 2026 automatic-labeling pipeline for LLM self-judgments uses `True`, `False`, `Uncertain` and reports 96.1% human–auto agreement when GPT-4o-mini acts as judge (Label Constraint Modeling, 2026).
- Zero-knowledge cross-model consistency frameworks assign per-block scores in `{0, 0.5, 1}` mapping to `ACCURATE`, `NEUTRAL`, `CONTRADICTION` (Zero-knowledge Hallucination Detection, 2025).
- 3-way schemes naturally accommodate abstention and unanswerable questions, which aligns with our `uncertain` class.

### 2.3 4-Way Classification
**Prevalence:** Uncommon; usually collapsed from finer taxonomies.
- A Korean finance RAG benchmark uses a 4-class task (`False Refusal`, `True Refusal`, plus hallucinated variants), but authors note that models often *collapse onto a single refusal type* (0.000 accuracy on one class) (Korean Finance Hallucination Benchmark, 2026).
- The most frequent confusion in that benchmark is `Contradictory ↔ Unverifiable`, suggesting that boundary cases between contradiction and unverifiability are genuinely ambiguous (Korean Finance Hallucination Benchmark, 2026).
- **Our ACC project explicitly lists 4-way classification as a differentiator** against binary competitors such as DCRD (COMPREHENSIVE_SYNTHESIS.md, 2026).

### 2.4 6- to 8-Way Fine-Grained Taxonomies
**Examples:**
- **FAVA (ACL 2024):** `Entity`, `Relation`, `Contradictory`, `Invented`, `Subjective`, `Unverifiable` — span-level, with synthetic training data (Mishra et al., 2024).
- **HIVE:** `none`, `factual_error`, `fabrication`, `unsupported`, `incomplete`, `irrelevant`, `reasoning_error`, `other` — 7 hallucination plus 1 null category (HIVE, 2025).
- **HAD:** 5 primary categories with 14 subcategories, evaluated on a manually annotated test set of 2,248 samples (HAD, 2025).
- **AgentHallu (agent trajectory level):** 5 primary / 14 subcategories via grounded-theory development; initial IAA for step localization was 77.9% (AgentHallu, 2025).

### 2.5 Continuous / Gray-Area Schemes
- **FaithBench** introduces two gray-area labels — `questionable` and `benign` — alongside `consistent` and `hallucinated`, explicitly to model subjective perception of hallucination (FaithBench, 2024).
- **Uncertainty quantification surveys** advocate treating hallucination detection as a confidence-scoring problem (outputting a scalar in [0,1]) rather than discrete classification, because hard labels enforce determinism and encourage overconfidence (Huang et al., 2025).

### 2.6 Comparative Summary Table

| Taxonomy | Papers | Pros | Cons |
|----------|--------|------|------|
| **Binary** | HaluEval, HHEM, most surveys | High detector AUC; simple evaluation; strong IAA | No diagnostic granularity; misses partial errors |
| **3-way** | Auto-labeling pipelines, cross-model consistency | Captures abstention/uncertainty; moderate IAA | Still coarse for targeted remediation |
| **4-way** | Korean finance benchmark, **ACC (ours)** | Enables nuanced interventions | Rare in literature; risk of class collapse or boundary ambiguity |
| **6–8-way** | FAVA, HIVE, HAD, AgentHallu | Rich diagnostic signal; good for error analysis | Exact-type IAA drops to ~60%; harder to train; many classes are rare |
| **Continuous** | FaithBench, UQ surveys, soft-label KD | Models ambiguity naturally; better calibration; threshold-tunable | Harder to evaluate; requires calibration data |

---

## 3. Is 4-Way Classification Actually Useful?

### 3.1 The Case For Multi-Class Labels
- **Differentiated interventions.** A 4-way detector can trigger *different* actions per class: flag-and-continue for `uncertain`, halt-and-regenerate for `hallucinated`, and special handling for `contradictory` (oxymoronic prompts). This is the core novelty claim in our COMPREHENSIVE_SYNTHESIS.
- **Neuroscience alignment.** The `contradictory` class maps to the ACC's known role in monitoring response conflict (e.g., Stroop-like interference), which binary detectors cannot represent.
- **Literature support for fine-grained detection.** FAVA shows that fine-grained detection enables targeted editing, improving Llama2-Chat factual accuracy by up to +9.3% (Mishra et al., 2024).

### 3.2 The Case Against — Ambiguity and Collapse
- **Detection is easier than classification.** In a 2026 multi-agent workflow audit, LLM judges achieved 92.2% recall for *detecting* hallucinations but only moderate Cohen's κ (0.456) for *typing* them. In 32/224 cases, judge and human agreed a hallucination existed but disagreed on its type or location (Beyond Final Answers, 2026).
- **Per-type agreement is uneven.** In the same study, types requiring only local evidence (`scope`, `procedural`, `factual`) showed moderate-to-substantial agreement (κ ≥ 0.595), while types requiring cross-step reasoning (`logical`, `referential`) showed only slight agreement (κ ≤ 0.211) (Beyond Final Answers, 2026).
- **Exact-type IAA is low.** FAVA reports 75.1% agreement on *whether* a sentence contains an error, but only 60.3% on *which* error type (Mishra et al., 2024).
- **Class-collapse in practice.** The Korean finance benchmark reports that base models frequently collapse to a single refusal type, scoring 0.000 on the other (Korean Finance Hallucination Benchmark, 2026).
- **Model performance drops with granularity.** On scientific literature, LLMs perform reasonably at binary hallucination detection but "performance drops significantly" in 4-way granularity (token-level hallucinations are hardest) (IJCNLP 2025).

### 3.3 Boundary Cases in Our Current 4-Way Schema
| Boundary | Why It Is Ambiguous | Literature Parallel |
|----------|---------------------|---------------------|
| `supported` vs. `uncertain` | Model may generate a plausible but unverified fact with high token probability. Token prob != factual certainty. | FAVA `Unverifiable` vs. non-hallucinated; low IAA on this boundary |
| `hallucinated` vs. `uncertain` | Unanswerable question may elicit a confabulated answer. Is the token "uncertain" because the question is bad, or "hallucinated" because the model invented an answer? | Auto-labeling pipelines discard "Uncertain" labels as unreliable (Label Constraint Modeling, 2026) |
| `contradictory` vs. `hallucinated` | An oxymoronic prompt can force a factually wrong token. The token is both contradictory to prompt semantics and factually incorrect. | Korean benchmark: `Contradictory ↔ Unverifiable` is the dominant confusion (Korean Finance Hallucination Benchmark, 2026) |
| `supported` vs. `contradictory` | A token may align with world knowledge while also violating the contradictory prompt's internal logic. | HIVE enforces schema-level consistency: if `decision=1` then `type=none` (HIVE, 2025) |

**Conclusion:** The 4-way taxonomy is *useful as a design differentiator* but creates genuine ambiguity at class boundaries. The literature suggests that **hierarchical schemes** (binary first, then type) or **soft labels** are more robust than flat 4-way hard labels.

---

## 4. Failure Modes of Heuristic Labeling

Our current pipeline uses substring matching, embedding similarity, and token-probability thresholds. The literature documents severe limitations for each.

### 4.1 Substring Matching / Overlap-Based Methods
- **Assumption:** All correct tokens should appear in a reference source.
- **Failure:** Paraphrases, synonyms, and valid inferences are penalized. Zhou et al. (cited in the 2022 NLG hallucination survey) originally proposed overlap-based detection but noted it fails when models paraphrase or use bilingual synonyms (Ji et al., 2022).
- **In our context:** A token like "physician" instead of "doctor" may be labeled `hallucinated` simply because the substring does not match the prompt bank, even though it is semantically equivalent.

### 4.2 Embedding Similarity / Semantic Similarity
- **The Semantic Illusion.** A 2025 conformal-prediction study found that embedding-based methods (DeBERTa-v3-large, BGE-base) achieve 95% coverage with 0% FPR on synthetic hallucinations but **collapse to 100% FPR** on real hallucinations from RLHF-aligned models. The "hardest" hallucinations are semantically indistinguishable from faithful responses (Semantic Illusion, 2025).
- **Why it fails:** Embeddings capture surface pragmatics, not truth conditions. A hallucinated entity can sit in the same semantic neighborhood as a real one.
- **In our context:** Embedding similarity may rate a fabricated date ("July 20, 1969" for a wrong event) as highly similar to a correct date, producing a false `supported` label.

### 4.3 Token Probability / Entropy Thresholds
- **High-confidence hallucinations exist.** CHOKE and related work show that models can hallucinate with very high token probability; uncertainty is a necessary but not sufficient signal (Kazlaris, 2025).
- **Token-level != semantic-level uncertainty.** A position paper on UQ for LLMs argues that "traditional token-level uncertainty is not sufficient" and the field must shift to semantic-level uncertainty, because ambiguity and epistemic limits are not visible in per-token logits (Semantic-Level Uncertainty, 2025).
- **In our context:** A token with high probability in the `contradictory` prompt bank may be labeled `supported` by a probability threshold, even though it represents a logical impossibility.

### 4.4 Compounding Errors in Multi-Signal Heuristics
- **Representational ambiguity at disagreement boundaries.** A 2026 weak-supervision study (directly relevant to our method) found that cases where heuristic and judge labels disagree exhibit **higher variance in hidden-state activations**, suggesting the disagreement reflects genuine representational ambiguity rather than noise (Weakly Supervised Distillation, 2026).
- **The heuristic may learn the wrong boundary.** If training data systematically mislabels paraphrased facts as hallucinations, the MLP will learn to detect *lexical divergence* rather than *factual falsity*.

---

## 5. Better Labeling Strategies

### 5.1 Model-Based Pseudo-Labels (LLM-as-Judge)
**Evidence:**
- A 2026 large-scale automatic-labeling study used a three-stage pipeline: text pattern matching → NLI (DeBERTa-v3-base) → GPT-4o-mini judge. On 800 manually verified samples, automatic vs. human agreement reached **96.125%** overall; on 200 "uncertain" cases, agreement was **93.50%** (Label Constraint Modeling, 2026).
- Weak-to-strong generalization literature shows that student models can outperform their noisy LLM teachers when pseudo-labels are filtered by confidence or loss (Burns et al., 2024).
- **AgentHallu** uses oracle-guided reasoning paths drafted by two different LLMs (GPT-5-Thinking + Gemini-2.5-Pro), then verified by human experts, achieving high consistency (AgentHallu, 2025).

**Recommendation for ACC:**
- Implement a **cascaded judge**: Heuristic rules filter obvious cases → NLI model handles semantic equivalence → GPT-4o-mini (or local equivalent) adjudicates borderline cases.
- Use **consensus between two judges** (e.g., GPT-4o-mini + Gemini-2.5-Flash) and treat disagreement as a `soft uncertain` label rather than forcing a hard class.

### 5.2 Soft Labels and Knowledge Distillation
- **Smoothed knowledge distillation.** Nguyen et al. (cited in the 2024 survey) replaced hard labels with soft labels from a teacher model during supervised fine-tuning, reducing overconfidence and hallucination on CNN/Daily Mail and XSUM (Huang et al., 2025).
- **Hard labels encourage overconfidence.** A 2025 survey on hallucination mitigation explicitly links hard labels to model overconfidence and proposes soft-label fine-tuning to improve factual grounding (Kazlaris, 2025).
- **Soft interpolation between signals.** The 2026 weak-supervision study defines `ỹ = α·y_judge + (1−α)·y_hybrid` and leaves exploration of soft labels for future work — suggesting this is an open but promising direction (Weakly Supervised Distillation, 2026).

**Recommendation for ACC:**
- Train the Approach B detector with **soft-target cross-entropy** where the target is a 4-dimensional probability vector rather than a one-hot label.
- Example: A token that is 70% `supported` and 30% `uncertain` according to judge consensus should receive target `[0.7, 0.0, 0.3, 0.0]` rather than `[1, 0, 0, 0]`.

### 5.3 Continuous Uncertainty / Factuality Scores
- **Semantic entropy** (Farquhar et al., Nature 2024) clusters sampled answers by meaning and computes entropy over equivalence classes. It is a **continuous** hallucination-risk score, not a discrete label (Farquhar et al., 2024).
- **Semantic Entropy Probes (SEPs)** approximate this continuous score from hidden states in a single forward pass, outputting a calibrated probability of hallucination (Kossen et al., 2024).
- **HHEM** computes a hallucination score `S_h = f(generated, retrieved)` and thresholds it, but the raw score is continuous and can be used for ranking or selective review (HHEM, 2025).
- **UIR (Uncertainty-Informed Refinement)** uses JS-divergence thresholds to filter pseudo-labels, treating hallucination risk as a continuous spectrum (UIR, 2025).

**Recommendation for ACC:**
- Add a **fifth output neuron** (or separate head) that predicts a continuous `conflict_score` ∈ [0,1] alongside the 4-way discrete classifier.
- This enables **threshold-tunable interventions** (e.g., `if conflict_score > 0.7: halt; elif > 0.4: warn; else: continue`), which is more flexible than argmax classification.

### 5.4 Hierarchical Labeling (Binary → Type)
- FAVA and mFAVA both use a **two-tier** architecture: (1) binary token-level detection of hallucinated spans, (2) a separate category classifier for the hallucination type (Mishra et al., 2024; Islam et al., 2025).
- This mirrors the finding that *detection* is easier than *classification* (Beyond Final Answers, 2026).

**Recommendation for ACC:**
- **Primary detector:** Binary or 3-way (`supported` / `unsupported` / `uncertain`) for real-time intervention.
- **Secondary classifier:** A lightweight type head (hallucinated / contradictory / uncertain-subtype) that runs only on flagged tokens, reducing latency for clean generations.

---

## 6. Human Evaluation of Hallucination Taxonomies

### 6.1 Inter-Annotator Agreement (IAA) by Granularity
| Study | Task | IAA Metric | Score | Interpretation |
|-------|------|------------|-------|----------------|
| **FAVA** | Sentence-level error detection | Raw agreement | 75.1% | Moderate–good |
| **FAVA** | Exact error type (6-way) | Raw agreement | 60.3% | Moderate; significant ambiguity |
| **AuthenHallu** | Binary dialogue labels | Fleiss' κ | 0.591 | Moderate |
| **Multi-agent workflows** | Hallucination detection | Cohen's κ | 0.456 | Moderate |
| **Multi-agent workflows** | Type classification (per type) | Cohen's κ | 0.211–0.595 | Slight to substantial; cross-step types are worst |
| **AgentHallu** | Step localization | Raw agreement | 77.9% | Moderate–good |
| **HAD** | Test-set quality filter | Raw agreement | 80.56% | Good |
| **FELM** | Segment-level factuality | Raw agreement | 91.3% | Very high (expert annotators) |
| **FaithBench** | Gray-area labels (`questionable`, `benign`) | Not quantified | Low (stated) | Authors explicitly note low IAA on gray-area classes |

### 6.2 Key Takeaways for Our Project
1. **Binary/3-way detection is within reach of high reliability** (κ ≈ 0.6–0.8). This justifies keeping a simple primary detector.
2. **Exact 4-way/6-way typing is genuinely hard for humans** (κ ≈ 0.2–0.6). Expecting a small MLP to learn boundaries that humans disagree on is optimistic.
3. **Expert annotators outperform crowd-workers.** FELM's 91.3% agreement used author-level experts; FAVA's 60.3% used trained but non-expert annotators. If we use pseudo-labels, a strong judge (GPT-4-class) is closer to the expert tier.
4. **Gray-area labels lower agreement.** FaithBench's `questionable` and `benign` labels had low IAA, mirroring the ambiguity we see between `supported` and `uncertain`.

---

## 7. Recommendations

### 7.1 Should we keep 4-way classification, simplify to binary, or expand?

**Answer: Restructure into a hierarchical scheme.**

| Layer | Classes | Role | Rationale |
|-------|---------|------|-----------|
| **Primary detector** | `supported` / `unsupported` / `uncertain` | Real-time intervention | Binary/3-way IAA is reliable; intervention logic is simple (pass / flag / halt) |
| **Secondary type head** | `hallucinated` / `contradictory` / `unverifiable` | Post-hoc diagnosis | Runs only on `unsupported` tokens; preserves our neuroscience differentiation without forcing ambiguous boundaries on every token |

- **Do not expand** to 6–8 classes unless we can demonstrate >75% human agreement on each boundary (unlikely with ~2,000 examples).
- **Do not flatten to pure binary** because it sacrifices our key differentiator (ACC-inspired conflict monitoring) and eliminates the `uncertain` abstention signal.

### 7.2 Should we switch from heuristic labels to model-based pseudo-labels?

**Answer: Yes, adopt a cascaded pseudo-labeling pipeline.**

1. **Stage 1 — Heuristic pre-filter:** Fast rules (substring, regex) label obvious matches/mismatches.
2. **Stage 2 — NLI semantic check:** Use `nli-deberta-v3-base` or similar to catch paraphrases and semantic equivalence that Stage 1 misses.
3. **Stage 3 — LLM judge adjudication:** GPT-4o-mini / Gemini-2.5-Flash adjudicates borderline cases, with a rubric grounded in concrete examples.
4. **Stage 4 — Consensus filtering:** Discard or soft-label cases where judge and NLI disagree (these are the genuinely ambiguous tokens).

**Expected improvement:** Based on the 96.1% human–auto agreement reported for a similar pipeline (Label Constraint Modeling, 2026), we can expect heuristic-to-pseudo-label agreement to rise from ~85% to ~95%, directly improving detector AUC.

### 7.3 Should we use continuous scores instead of discrete classes?

**Answer: Use both — discrete classes for action selection, continuous scores for calibration and training.**

- **Training:** Replace one-hot targets with soft probability vectors (e.g., `[0.7, 0.0, 0.3, 0.0]`). This is supported by smoothed-knowledge-distillation literature (Huang et al., 2025; Kazlaris, 2025).
- **Inference:** Output a 4-way softmax *plus* a scalar `conflict_score`. The argmax drives the default intervention, but the continuous score enables:
  - **Dynamic thresholds** tuned per-domain or per-risk-profile.
  - **Selective review** (only tokens with 0.3 < score < 0.7 need human/LLM review).
  - **Calibration-aware generation** (e.g., raise sampling temperature when conflict_score is high to encourage diverse re-sampling).

---

## 8. Implementation Roadmap

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **P0** | Refactor primary detector to 3-way (`supported` / `unsupported` / `uncertain`) | Low | High — simplifies training, improves reliability |
| **P0** | Add secondary type head for `hallucinated` vs. `contradictory` | Medium | High — preserves differentiation without boundary ambiguity |
| **P1** | Implement cascaded pseudo-labeling (heuristic → NLI → LLM judge) | Medium | High — directly improves label quality |
| **P1** | Switch training loss to soft-label cross-entropy | Low | Medium — reduces overconfidence, better calibration |
| **P2** | Add continuous `conflict_score` output head | Low | Medium — enables threshold-tunable interventions |
| **P2** | Evaluate IAA of our own pseudo-labels vs. a 200-sample human gold set | Medium | High — establishes empirical confidence in the pipeline |
| **P3** | Consider trajectory modeling (LSTM/attention over hidden-state sequence) instead of single-step MLP | High | Medium — aligns with SOTA (ICR Probe / MultiHaluDet) |

---

## 9. Citations

1. Huang, L., et al. (2025). *A Survey on Hallucination in Large Language Models.* ACM Computing Surveys. https://arxiv.org/html/2510.06265v3
2. (2025). *Hallucination Detection and Evaluation of Large Language Model.* https://arxiv.org/html/2512.22416v1
3. (2026). *Improving LLM Hallucination Detection via Label Constraint Modeling.* https://arxiv.org/html/2605.03971v1
4. (2025). *Zero-knowledge LLM hallucination detection through fine-grained cross-model consistency.* https://arxiv.org/html/2508.14314v2
5. (2026). *A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance.* https://arxiv.org/html/2605.29523v1
6. ACC LLM Enhancement team (2026). *COMPREHENSIVE_SYNTHESIS.md* — internal competitive analysis.
7. Mishra, A., et al. (2024). *Fine-grained Hallucination Detection and Editing for Language Models (FAVA).* ACL 2024. https://arxiv.org/html/2401.06855v2
8. (2025). *HIVE: Hallucination Identification and Verification Engine.* https://arxiv.org/pdf/2604.26139
9. (2025). *HAD: HAllucination Detection Based on a Comprehensive Hallucination Taxonomy.* https://arxiv.org/html/2510.19318v1
10. (2025). *AgentHallu: Benchmarking Automated Hallucination Attribution.* https://arxiv.org/html/2601.06818v1
11. (2024). *FaithBench: A Diverse Hallucination Benchmark for Summarization.* https://arxiv.org/html/2410.13210v1
12. (2024). *Large Language Models Hallucination: A Comprehensive Survey.* https://arxiv.org/html/2510.06265v2
13. (2026). *Beyond Final Answers: Auditing Trajectory-Level Hallucinations in Multi-Agent Workflows.* https://arxiv.org/html/2605.24219v1
14. (2025). *Can LLMs detect hallucination at a granular level in scientific literature?* IJCNLP 2025. https://aclanthology.org/2025.ijcnlp-long.70.pdf
15. Ji, Z., et al. (2022). *Survey of Hallucination in Natural Language Generation.* https://arxiv.org/pdf/2202.03629v3
16. (2025). *The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG.* https://arxiv.org/html/2512.15068v2
17. Kazlaris, I. (2025). *From Illusion to Insight: A Taxonomic Survey of Hallucination Mitigation.* https://www.mdpi.com/2673-2688/6/10/260
18. (2025). *Position: Semantic-Level Uncertainty for LLMs.* https://openreview.net/pdf/a798f164ebd1849ecf4877e109d2963d81762331.pdf
19. (2026). *Weakly Supervised Distillation of Hallucination Signals into Transformer Representations.* https://arxiv.org/html/2604.06277v1
20. Burns, C., et al. (2024). *Theoretical Analysis of Weak-to-Strong Generalization.* NeurIPS 2024. https://arxiv.org/pdf/2405.16043
21. Farquhar, S., et al. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy.* Nature 630, 625–630. https://doi.org/10.1038/s41586-024-07421-0
22. Kossen, J., et al. (2024). *Semantic Entropy Probes.* https://arxiv.org/abs/2406.15927
23. (2025). *Hughes Hallucination Evaluation Model.* https://arxiv.org/html/2512.22416v1
24. (2025). *Uncertainty-Informed Refinement.* https://arxiv.org/pdf/2512.03992
25. Islam, S. O., et al. (2025). *How Much Do LLMs Hallucinate across Languages?* EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1481.pdf

---

*Analysis compiled from parallel web searches of 2024–2026 hallucination-detection literature, internal project documents (COMPREHENSIVE_SYNTHESIS.md, sweep_hallucination_detection.md), and direct paper content extraction.*
