# ACC LLM Enhancement — Comprehensive Research Audit & Slide Source Document

**Date:** May 28, 2026  
**Auditor:** Telamon (AI Research Assistant)  
**Project:** Anterior Cingulate Cortex (ACC) Inspired Hallucination Detection for Large Language Models  
**Affiliation:** AUB Research  
**Hardware:** NVIDIA RTX 3080 10GB (Desktop) + Jetson Orin Nano (Edge)  
**Paper Target:** Novel inference-time verification layer inspired by neuroscience, with QLoRA fine-tuning for domain adaptation

---

## Executive Summary

ACC LLM is a **two-pronged research effort**: (1) a neuroscience-inspired inference-time verification layer that detects hallucinations and contradictions in LLM outputs by monitoring per-token entropy and hidden-state conflict patterns, and (2) a QLoRA fine-tuning pipeline for domain-specific adaptation. The **concept is novel and well-motivated**, the **code architecture is implemented**, and the **training pipeline is verified end-to-end**. However, **the entropy computation has a critical bug** (always returns 0), **no real model training has occurred** (Mistral weights incomplete), and **the dataset is generic Alpaca rather than clinical**.

| Supervisor-Readiness Rating | **🟡 PROMISING — Strong Concept, Weak Validation** |
|---|---|
| Can show novel idea | ✅ Yes — neuroscience → AI is a compelling angle |
| Can show implemented code | ✅ Yes — 3 modules implemented, pipeline verified |
| Can show working verification | ❌ No — entropy bug prevents detection |
| Can show fine-tuned model | ❌ No — Mistral shards incomplete, no training |
| Ready for AAAI submission | ❌ No — needs 2-4 weeks for bug fix + real training + dataset upgrade |

---

## 1. Problem Statement & Motivation

### The Hallucination Crisis
Large Language Models (LLMs) generate fluent, authoritative-sounding text that is frequently **factually wrong, contradictory, or hallucinated**. In high-stakes domains (medicine, law, science), this is catastrophic.

| Domain | Hallucination Risk | Example |
|---|---|---|
| **Clinical** | Patient harm | GPT-4 invents drug interactions, recommends unsafe dosages |
| **Legal** | Liability | Cites non-existent cases, misinterprets statutes |
| **Scientific** | Research waste | Generates fake citations, fabricated experimental details |

### Existing Solutions (and Their Limits)

| Approach | Examples | Limitation |
|---|---|---|
| **Retrieval-Augmented Generation (RAG)** | LangChain, LlamaIndex | Only prevents hallucinations *about known documents*; doesn't catch contradictions within the generated text itself |
| **Self-Consistency Sampling** | Wang et al., ICLR 2023 | Expensive (N× inference cost); doesn't provide per-token uncertainty |
| **Logit-based Uncertainty** | Kadavath et al., 2022; Lin et al., 2023 | Uses token probability as proxy for confidence — but doesn't model *conflict* or *contradiction* |
| **Fact-Checking Pipelines** | Min et al., ACL 2023 | Post-hoc: detects errors after generation, doesn't prevent them |

### The Neuroscience Insight
The **Anterior Cingulate Cortex (ACC)** in the human brain plays a central role in:
- **Conflict monitoring**: Detecting when expected and actual outcomes disagree
- **Error detection**: Signaling when cognitive control needs to increase
- **Uncertainty encoding**: Representing the reliability of predictions

**Key paper:** Botvinick et al. (2001), "Conflict monitoring and anterior cingulate cortex: an update." *Cognitive, Affective, & Behavioral Neuroscience*.

**Our hypothesis:** We can build an analogous computational module for LLMs — a per-token "ACC layer" that monitors the model's own confidence and signals when additional verification is needed.

---

## 2. Literature Review

### 2.1 LLM Hallucination Detection

**SelfCheckGPT (Manakul et al., 2023)**  
Uses multiple sampled responses and checks consistency against a reference. Post-hoc and expensive.

**LM vs. LM (Cohen et al., 2023)**  
Uses a second LLM to fact-check the first. Suffers from the same hallucination risk in the checker.

**FactScore (Min et al., ACL 2023)**  
Decomposes generation into atomic facts and verifies each against a knowledge source. High precision but high latency.

**Semantic Entropy (Kadavath et al., 2022; Kuhn et al., NeurIPS 2023)**  
Clustering generated responses by semantic meaning to estimate uncertainty at the *sequence* level. Does not provide per-token resolution.

**LogitLens (nostalgebraist, 2020; Belrose et al., 2023)**  
Projects intermediate hidden states through the output embedding to read "what the model is thinking" at each layer. Inspired our hidden-state conflict detector.

### 2.2 Neuroscience-Inspired AI

**Predictive Processing / Free Energy Principle (Friston, 2010)**  
The brain as an inference machine that minimizes prediction error. Our entropy monitor mirrors prediction-error signaling.

**Conflict Monitoring Theory (Botvinick et al., 2001; 2004)**  
ACC as a conflict detector. Our sliding-window entropy + threshold breach mechanism is a direct computational analog.

**Neuromorphic Computing (Davies et al., 2018; Intel Loihi)**  
Hardware that mimics neural architectures. We are not building neuromorphic hardware but borrowing functional principles.

**NeuroAI (Zador et al., 2023; Nature Neuroscience)**  
Emerging field advocating that neuroscience insights should guide AI architecture design. Our work is a concrete instantiation of this philosophy.

### 2.3 Efficient LLM Fine-Tuning

**LoRA (Hu et al., ICLR 2022)**  
Low-Rank Adaptation: freeze base weights, train low-rank decomposition matrices. Reduces trainable params by 10,000×.

**QLoRA (Dettmers et al., NeurIPS 2023)**  
4-bit quantization + LoRA + paged optimizers. Enables fine-tuning 65B models on single 48GB GPU. Our desktop config adapts QLoRA for 10GB.

**DoRA (Liu et al., 2024)**  
Weight-Decomposed Low-Rank Adaptation. Further improves on LoRA. Could be incorporated in future iterations.

### 2.4 Medical LLMs

**Med-PaLM 2 (Singhal et al., 2023)**  
PaLM 2 fine-tuned on medical QA. Achieves >85% on USMLE but still hallucinates.

**PMC-LLaMA (Wu et al., 2023)**  
LLaMA fine-tuned on 4.8M biomedical papers. Domain-adapted but without explicit hallucination guardrails.

**GatorTron (Peng et al., 2022)**  
Clinical NLP model from UF Health. Demonstrates value of clinical-domain pretraining.

**Huatuo (Zhang et al., 2023)**  
Chinese medical LLM with RAG. Shows that domain adaptation + retrieval helps but doesn't eliminate hallucinations.

**Gap:** None of these systems have an explicit, neuroscience-inspired *verification layer* that monitors the model's own confidence during generation. They all rely on post-hoc fact-checking or RAG.

---

## 3. Architecture & Methodology

### 3.1 Overview: Two Approaches

```
┌─────────────────────────────────────────────────────────┐
│                 ACC LLM Enhancement                       │
├─────────────────────────────────────────────────────────┤
│  Domain Adaptation          │  Inference Verification   │
│  (Training Time)            │  (Generation Time)          │
│                             │                             │
│  QLoRA Fine-Tuning          │  Approach A: Entropy      │
│  ─ Mistral 7B 4-bit        │     Monitor (Logits)        │
│  ─ Medical dataset          │     ─ per-token H           │
│  ─ LoRA r=32, α=64         │     ─ sliding window        │
│                             │     ─ threshold breach      │
│                             │                             │
│                             │  Approach B: Conflict       │
│                             │     Detector (Hidden States)│
│                             │     ─ MLP on layer -4       │
│                             │     ─ 4-way classifier      │
│                             │                             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Approach A: Entropy-Based Uncertainty Monitoring (Logits)

**Inspiration:** ACC conflict signal is analog and graded — not binary. Per-token entropy is the computational equivalent.

**Components:**

#### `EntropyMonitor`
```python
class EntropyMonitor:
    """
    - Computes per-token Shannon entropy: H = -Σ p(x) log p(x)
    - Three threshold modes:
      1. absolute:    H > fixed threshold (e.g., 3.5 nats)
      2. moving_average: H > μ_window + k·σ_window  (adaptive)
      3. percentile:    H > P95(window)  (robust to outliers)
    - Actions on breach:
      1. flag:     insert [UNCERTAIN] marker
      2. regenerate: re-sample at higher temperature
      3. warning:  prefix with warning string
    - Confidence score: normalized 0-1 based on breach frequency
    """
```

#### `_EntropyLogitsProcessor` (HuggingFace Integration)
```python
class _EntropyLogitsProcessor(LogitsProcessor):
    """
    Injected into transformers .generate() pipeline.
    Called after model forward, before token selection.
    Observes raw logits → computes entropy → optionally intervenes.
    """
```

#### `ACCEnhancedGenerator`
```python
class ACCEnhancedGenerator:
    """
    Wrapper around any HuggingFace causal LM.
    Usage:
        gen = ACCEnhancedGenerator(model, tokenizer, monitor)
        output = gen.generate(prompt, use_acc=True)
        # Returns: text + per_token_entropy + uncertain_steps + confidence_score
    """
```

**Current Status:** ⚠️ **Entropy computation bug** — `_EntropyLogitsProcessor.observe()` always returns 0.0. Likely cause: receiving post-softmax logits instead of pre-softmax, or softmax applied twice.

### 3.3 Approach B: Latent-State Conflict Detector (Hidden States)

**Inspiration:** ACC receives input from multiple cortical areas. Similarly, intermediate hidden states encode "what the model is thinking" before output projection.

**Components:**

#### `HiddenStateExtractor`
```python
class HiddenStateExtractor:
    """
    Registers a forward hook on a specified transformer layer.
    Extracts hidden states during generation for downstream classification.
    """
```

#### `LatentConflictDetector` (Small MLP Classifier)
```python
class LatentConflictDetector(nn.Module):
    """
    ~100K parameters.
    Input: hidden states from layer -4 (batch, seq_len, hidden_dim)
    Output: 4-class logits (supported, hallucinated, uncertain, contradictory)
    
    Architecture: Linear → LayerNorm → ReLU → Dropout → Linear → 4-way
    """
```

**Training Strategy:**
- Generate synthetic conflict data (contradictory pairs, hallucinated facts)
- Extract hidden states during generation
- Train detector as 4-way classifier
- Freeze LLM, train only detector

**Current Status:** ✅ Architecture implemented, training script written, no training run yet (blocked by model availability).

### 3.4 Domain Adaptation: QLoRA Fine-Tuning

**Model:** Mistral 7B Instruct v0.3  
**Quantization:** 4-bit NF4 (Dettmers et al., 2023)  
**LoRA Config:** r=32, α=64, dropout=0.05  
**Target Modules:** q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj  
**Optimizer:** paged_adamw_8bit  
**Training:** 3 epochs, batch=1, accum=4, max_seq_length=1024  

**Pipeline Verified:** ✅ Smoke test with tiny GPT-2 (1.1M params) passed:
- Model loading ✅
- LoRA attachment ✅
- SFTTrainer loop ✅
- Gradient checkpointing ✅
- Checkpoint saving ✅
- Loss decreased: 5.47 → 3.76 ✅

**Current Blocker:** Mistral 7B weights incomplete (only shard 00001 of 00003 downloaded).

### 3.5 Dataset

**Current:** `data/medical/train.jsonl` — 700 samples from Alpaca health Q&A  
**Issues:**
- Generic health advice, not clinical
- No adversarial/hallucinated examples for detector training
- Too small for meaningful domain adaptation

**Target Datasets:**
| Dataset | Size | Domain | Source |
|---|---|---|---|
| PubMedQA | 1K | Biomedical QA | Jin et al., 2019 |
| MedQA | 61K | USMLE questions | Jin et al., 2021 |
| MIMIC-III | 58K | Clinical notes | Johnson et al., 2016 |
| MedAlign | 1K | Clinician-patient | Fleming et al., 2023 |

---

## 4. Experimental Results

### 4.1 Pipeline Verification (Tiny GPT-2)

| Component | Status | Result |
|---|---|---|
| Model loading | ✅ | `AutoModelForCausalLM.from_pretrained` |
| LoRA attachment | ✅ | `get_peft_model` with `LoraConfig` |
| TRL SFTTrainer | ✅ | 3 epochs completed |
| Gradient checkpointing | ✅ | Memory stable |
| Checkpoint saving | ✅ | `final_adapter/` written |
| Train loss trend | ✅ | 5.47 → 3.76 (31% decrease) |

**Trainable params:** ~0.3% of total (LoRA rank 8 on 1.1M param model)

### 4.2 ACC Validation Attempt (Tiny GPT-2 + ACC Layer)

| Metric | Expected | Actual | Status |
|---|---|---|---|
| Per-token entropy | > 0 for most tokens | 0.0 for ALL tokens | ❌ BROKEN |
| Confidence score | 0.0-1.0 | 1.0 (stuck) | ❌ BROKEN |
| Uncertain steps | Some flagged | None | ❌ BROKEN |
| Generated text | Coherent (post-adapter) | Garbage (tiny model) | ⚠️ Expected |

**Root cause hypothesis:** `_EntropyLogitsProcessor` receives logits *after* temperature scaling or top-k filtering, rather than raw pre-softmax logits. The `LogitsProcessor` API in transformers may pass processed logits.

### 4.3 Model Download Status

| Shard | Size | Status |
|---|---|---|
| model-00001-of-00003 | 4.61 GB | ✅ Complete |
| model-00002-of-00003 | ~0.03 GB | ⏳ Downloading |
| model-00003-of-00003 | ~8.86 GB | ⏳ Downloading |

**ETA:** 2-4 hours if connection stable (HuggingFace throttles unauthenticated downloads to ~150 kB/s).

---

## 5. Honest Assessment

### 5.1 What is Real & Demonstrable

✅ **Novel concept is well-articulated:** Neuroscience → AI is a strong, timely angle  
✅ **Code architecture is complete:** 3 modules (entropy, conflict detector, integration) implemented  
✅ **Training pipeline is verified:** QLoRA + TRL works end-to-end  
✅ **Dual-hardware strategy:** Desktop (10GB) + Jetson Orin Nano (edge)  
✅ **Integration with HuggingFace is clean:** `LogitsProcessor` approach is robust  

### 5.2 What is Missing or Broken

❌ **Entropy bug prevents any actual verification:** The core Approach A is non-functional  
❌ **No real model training:** Mistral weights incomplete  
❌ **No real dataset:** Generic Alpaca, not clinical  
❌ **Approach B untrained:** Conflict detector architecture exists but no training data or run  
❌ **No quantitative evaluation:** No F1, precision, recall for hallucination detection  
❌ **No comparison to baselines:** Cannot claim superiority over SelfCheckGPT, etc.  
❌ **No ablation study:** Is the ACC layer actually helping? Unknown.  

### 5.3 Supervisor Conversation Strategy

**Frame as "novel research proposal with strong preliminary implementation":**

> "We're developing a neuroscience-inspired verification layer for LLMs, inspired by the anterior cingulate cortex's role in conflict detection. The concept is novel — no existing system uses per-token entropy monitoring with adaptive thresholds during generation. We've implemented the full architecture and verified the training pipeline on a small model. We're now debugging the entropy computation and preparing for real training on Mistral 7B with clinical data."

**Do NOT claim:**
- "We can detect hallucinations" — the entropy layer is broken
- "We've fine-tuned Mistral" — no weights, no training
- "We evaluated on medical tasks" — dataset is generic Alpaca

**DO claim:**
- "The neuroscience angle is timely — NeuroAI is an emerging field (Zador et al., 2023)"
- "We've built a complete, modular codebase with clean APIs"
- "The QLoRA pipeline is verified and ready to scale"
- "We have a clear roadmap: fix entropy → download Mistral → train on PubMedQA → evaluate"

---

## 6. Upcoming Week Plan

### Priority 1: Fix Entropy Computation Bug (1-2 days)
- [ ] Debug `_EntropyLogitsProcessor.observe()` — verify raw vs. processed logits
- [ ] Test with known-high-entropy inputs (random logits should give max entropy)
- [ ] Verify on CPU with tiny model first, then GPU
- [ ] Success metric: entropy > 0 for diverse prompts, varies meaningfully across tokens

### Priority 2: Complete Mistral Download & First Training Run (2-3 days)
- [ ] Wait for / retry download of shards 00002 + 00003
- [ ] Run `validate_model_load.py` to verify full model loads correctly
- [ ] Launch first QLoRA training on medical Alpaca dataset (3 epochs, ~6-8 hours)
- [ ] Save adapter checkpoints and training log

### Priority 3: Upgrade Dataset to Clinical (2-3 days)
- [ ] Download PubMedQA or MedQA
- [ ] Reformat to Alpaca-style instruction-response
- [ ] Add adversarial examples (hallucinated answers labeled as such)
- [ ] Re-run training with clinical data

### Priority 4: Implement & Train Approach B (3-4 days)
- [ ] Generate synthetic conflict training data (1000+ examples)
- [ ] Extract hidden states from Mistral during generation
- [ ] Train `LatentConflictDetector` (100K params, fast)
- [ ] Evaluate 4-way classification accuracy

### Priority 5: Paper Positioning (ongoing)
- [ ] Draft abstract emphasizing NeuroAI angle
- [ ] Position against SelfCheckGPT, Semantic Entropy, FactScore
- [ ] Define evaluation protocol: hallucination detection F1 on known-bad outputs

---

## 7. Slide Deck Content Guide

### Slide 1-3: Title & The Hallucination Problem
- Title: "ACC LLM: A Neuroscience-Inspired Verification Layer for Language Models"
- Hallucination examples (medical, legal, scientific)
- The gap: existing solutions are post-hoc or expensive
- Neuroscience insight: the ACC detects conflict in real time

### Slide 4-7: Literature Review
- LLM hallucination detection landscape (SelfCheckGPT, FactScore, Semantic Entropy)
- Neuroscience: Botvinick et al. conflict monitoring theory
- NeuroAI movement (Zador et al., 2023)
- Efficient fine-tuning: LoRA, QLoRA (Hu et al., Dettmers et al.)
- Medical LLMs (Med-PaLM, PMC-LLaMA, Huatuo)

### Slide 8-12: Architecture
- Full system diagram (dual approach: entropy + conflict detector)
- Approach A: EntropyMonitor internals (threshold modes, actions, confidence score)
- Approach B: HiddenStateExtractor + LatentConflictDetector (layer -4, 4-way classifier)
- QLoRA config diagram (4-bit quantization, LoRA adapters, target modules)
- Integration: `LogitsProcessor` in HuggingFace pipeline

### Slide 13-15: Implementation Progress
- Code modules table (what's implemented, lines of code, test coverage)
- Pipeline verification results (tiny GPT-2 smoke test)
- Current blockers: entropy bug, model download, dataset
- **Be transparent — this is a proposal, not a results paper**

### Slide 16-18: Novelty & Positioning
- "No existing system uses per-token entropy with adaptive thresholds during generation"
- Comparison table: ACC LLM vs. SelfCheckGPT vs. Semantic Entropy vs. RAG
- NeuroAI positioning: concrete instantiation of neuroscience → AI

### Slide 19-21: What's Next
- Fix entropy bug (with debugging strategy)
- Complete Mistral download + first training run
- Dataset upgrade to PubMedQA/MedQA
- Evaluation protocol definition

### Slide 22-24: Impact & Vision
- Medical LLMs with built-in hallucination guardrails
- Edge deployment on Jetson Orin Nano (clinical settings without cloud)
- Open-source framework for domain-specific safe LLMs

---

## 8. Risk Assessment for Supervisor Meeting

| Risk | Likelihood | Mitigation |
|---|---|---|
| Supervisor: "This is too speculative" | Medium | Emphasize NeuroAI as timely; cite Zador et al. (Nature Neuroscience 2023) |
| Supervisor: "Where are the results?" | High | Be upfront: "This is a proposal-phase project with architecture complete but evaluation pending" |
| Supervisor: "Has anyone done entropy monitoring before?" | Medium | Acknowledge Kadavath et al. use token probability; our contribution is *adaptive thresholds + conflict metaphor* |
| Supervisor: "Can you get clinical data?" | Medium | PubMedQA and MedQA are public; MIMIC-III requires credentialing (~2 weeks) |
| Supervisor wants to drop ACC LLM for NephroTwin | Medium | NephroTwin is more complete; position ACC LLM as "long-term moonshot" |

---

## 9. Bottom Line

ACC LLM is the **most conceptually novel** of the three projects but the **least experimentally validated**. It has:
- A compelling neuroscience-inspired angle
- A complete, modular code architecture
- A verified training pipeline
- But: zero working verification results, no real model, no real data

**For AAAI:** This project is currently **not ready** for submission. It needs:
1. Entropy bug fix (1-2 days)
2. Real model training on Mistral 7B (2-3 days)
3. Clinical dataset + evaluation (1 week)
4. Comparison to SelfCheckGPT baseline (3-4 days)

**Total: ~2-3 weeks of focused work** to reach a AAAI-ready state. If the supervisor assigns dedicated time, this is achievable. Otherwise, it's better positioned as a **long-term research direction** or a **workshop paper** (e.g., NeurIPS Workshop on NeuroAI, or a medical AI workshop).

**Recommendation:** Present ACC LLM as the **aspirational third project** — the "big idea" that needs nurturing. Lead with NephroTwin (most complete), then 2VitSegDet (engineering rigor), then ACC LLM (visionary concept).
