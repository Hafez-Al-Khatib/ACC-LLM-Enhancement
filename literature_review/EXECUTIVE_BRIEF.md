# ACC LLM: Executive Brief — Literature Positioning & Strategic Plan

**Date:** 2026-05-30  
**Status:** 3 of 4 scouts complete + supplementary searches  
**Papers reviewed:** 60+ sources  
**Papers downloaded:** 8 core papers

---

## TL;DR

Our **individual components** (entropy monitoring, self-consistency, hidden-state detector) now have strong published competitors. **However, the integration of all three within an ACC-inspired architecture — plus multi-vertical evaluation and edge deployment — remains unique.**

**Critical pivot:** Reframe from "we propose three techniques" to "we propose a neuroscience-inspired verification layer that coordinates entropy, consistency, and hidden-state detection as **coordinated modules in a single generation pass.**"

**Immediate must-dos:** Add COIECD and Single-Pass HalluDet as baselines. Add NQ-Swap and HaluEval datasets. Clarify differentiation in abstract.

---

## 1. Direct Competitors by Component

### 🔴 Entropy Monitoring: COIECD (ACL Findings 2024)
- **What:** Token-level entropy band detection for knowledge conflicts
- **How:** Contrastive decoding (2× compute) — compares entropy with/without context
- **Our edge:** Single-pass, no retrieval required, calibrated threshold

### 🔴 Hidden-State Detector: DCRD (2026) + Single-Pass HalluDet (NeurIPS 2025)
- **What:** Lightweight MLP on hidden states for real-time detection
- **How:** Single-layer MLP (DCRD) or 3-layer MLP (HalluDet) on generation states
- **Our edge:** 4-way classification (supported/hallucinated/uncertain/contradictory) vs. their binary

### 🟡 Self-Consistency: Semantic Entropy (Nature 2024)
- **What:** Gold-standard post-hoc uncertainty via NLI clustering
- **How:** N samples + semantic equivalence grouping
- **Our edge:** Generation-time (not post-hoc), hidden-state embedding (no extra model)

### 🟡 Neuroscience Framing: MAP (Nature Comm. 2025) + COCO (2026)
- **What:** Brain-inspired conflict monitoring modules
- **How:** MAP for planning; COCO for neuron identification
- **Our edge:** Only trained ACC module for hallucination detection

---

## 2. What Makes Us Unique (Defensible Claims)

| Claim | Defensibility | Evidence Needed |
|-------|--------------|----------------|
| **Three-signal integration in single pass** | Strong | Ablation study showing F1 improvements over each signal alone |
| **ACC-inspired trained module** | Strong | Citation to MAP/COCO showing they don't train detectors |
| **4-way latent conflict classification** | Moderate | Show binary detectors miss nuanced cases |
| **Multi-vertical + edge evaluation** | Strong | Most papers test single domain on A100 |
| **Empirical entropy calibration** | Moderate | Show calibrated > fixed threshold on held-out data |

---

## 3. What We Must Add (Non-Negotiable)

### Baselines
1. **COIECD** — direct entropy competitor
2. **Single-Pass HalluDet** — direct hidden-state competitor
3. **SAPLMA** — classic hidden-state probe
4. **Semantic Entropy** — gold-standard post-hoc reference

### Datasets
1. **NQ-Swap** — standard for conflict detection (used by COIECD/DCRD)
2. **HaluEval** — standard hallucination benchmark
3. **TruthfulQA** — adversarial factual testing

### Ablations
1. Entropy-only vs. Consistency-only vs. Detector-only vs. Full
2. Prompt-encoding hidden states vs. generation-time hidden states
3. Fixed threshold vs. calibrated threshold
4. r=8 (Jetson) vs. r=32 (Desktop)

---

## 4. Technology Upgrades to Consider

| Upgrade | Source | Effort | Impact |
|---------|--------|--------|--------|
| **DoRA + PiSSA** instead of vanilla LoRA | ICML 2024 / NeurIPS 2024 | Low | Better stability, faster convergence |
| **AWQ** instead of bitsandbytes NF4 | MLSys 2024 (Best Paper) | Medium | Better quality at same bitrate |
| **Hybrid RAG + QLoRA** | Emerging best practice 2024 | Medium | Strongest factual accuracy |
| **Speculative decoding** on Jetson | IEEE TMC 2024/2025 | Medium | 2.9–9.3× speedup |
| **TIES adapter merging** | NeurIPS 2023 | Low | Merge multiple domain adapters cleanly |

---

## 5. Revised Paper Narrative

### Problem Statement (Keep)
LLMs hallucinate. Existing detection is post-hoc (SelfCheckGPT, FactScore) or requires retrieval (RAG). We need real-time, generation-time verification.

### Our Solution (Revise)
> "We present ACC LLM, a neuroscience-inspired inference-time verification layer that coordinates three complementary signals — calibrated entropy monitoring, semantic self-consistency, and a generation-time latent conflict detector — within a single generation pass. Unlike COIECD, which requires contrastive decoding doubling compute, or DCRD, which uses binary hidden-state classification, our ACC layer integrates entropy, consistency, and hidden-state geometry as coordinated modules trained to detect supported, hallucinated, uncertain, and contradictory content."

### Key Differentiators (Emphasize)
1. **Single-pass integration** — no contrastive decoding, no post-hoc sampling
2. **Four-way classification** — nuanced detection beyond binary
3. **Neuroscience grounding** — explicit ACC conflict-monitoring architecture
4. **Hardware-conscious** — validated on RTX 3080 and Jetson Orin Nano
5. **Multi-vertical** — medical, STEM, financial, general

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Reviewer says "COIECD already does entropy" | High | High | Pre-empt in abstract: "single-pass, not contrastive" |
| Reviewer says "DCRD already does hidden-state detection" | High | High | Pre-empt: "4-way vs. binary classification" |
| Reviewer questions neuroscience framing as "marketing" | Medium | Medium | Cite Webb MAP, COCO; show explicit module design |
| Baselines underperform because of tiny training data | Medium | High | Use combined 12K dataset; consider data augmentation |
| Jetson experiments fail due to OOM | Medium | Medium | Test with r=8, seq=512 first; have fallback config |

---

## 7. Next 48-Hour Action Plan

| Day | Task | Owner |
|-----|------|-------|
| 1 | Download Mistral 7B weights (in progress) | You |
| 1 | Revise paper abstract + introduction | Me |
| 1 | Implement COIECD-style baseline | Me |
| 2 | Run QLoRA training on combined dataset | You/Me |
| 2 | Generate conflict detector data on Mistral | Me |
| 2 | Add NQ-Swap and HaluEval to datasets | Me |
| 2 | Train conflict detector | Me |
| 2 | Run first ablation (entropy-only vs. full) | Me |

---

## 8. Files Available

```
literature_review/
├── sweep_entropy_decoding.md          # 20+ entropy/decoding papers
├── sweep_neuroscience_ai.md           # 20+ neuroscience/AI papers
├── sweep_efficient_finetuning.md      # 20+ PEFT/quantization papers
├── COMPREHENSIVE_SYNTHESIS.md         # Full competitive analysis
├── EXECUTIVE_BRIEF.md                 # This file
├── synthesis_framework.md             # Template for ongoing analysis
└── papers/
    ├── farquhar2024_semantic_entropy_nature.pdf
    ├── zhang2024_edt_entropy_dynamic_temperature.pdf
    ├── chen2025_uncertainty_quantification_survey.pdf
    ├── han2025_finegrained_confidence_generation.pdf
    ├── labrak2024_biomistral_medical_llm.pdf
    ├── kuhn2023_semantic_entropy.pdf
    ├── yuan2024_coiecd_entropy_conflict_decoding.pdf
    └── obeso2025_realtime_entity_hallucination.pdf
```

---

*One scout (hallucination detection) still pending — will append findings when complete.*
*Last updated: 2026-05-30*
