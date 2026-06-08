# Literature Synthesis Framework — ACC LLM Positioning

**Date:** 2026-05-30  
**Purpose:** Map our project against current SOTA to identify gaps, threats, and opportunities

---

## 1. Competitive Landscape Matrix

### 1.1 Hallucination Detection Methods

| Method | When It Runs | Signals Used | External Dependencies | Latency | Our Overlap |
|--------|-------------|--------------|----------------------|---------|-------------|
| **Semantic Entropy (Farquhar Nature 2024)** | Post-hoc | Multiple samples + NLI | Sentence transformer | High (N generations) | We use similar embedding clustering but at generation time |
| **SAPLMA (Azaria 2023)** | Post-hoc | Single hidden state | None | Low | We extract hidden states at generation time, not prompt encoding |
| **SEPs (Kossen ICML 2024)** | Post-hoc | Hidden state → semantic entropy | None | Low | Similar probe idea but we predict 4 classes, not entropy |
| **SelfCheckGPT (Miao 2023)** | Post-hoc | Multiple samples + BLEU/NLI | None | High | Our self-consistency is similar but integrated in generate() |
| **SNNE (Nguyen ACL 2025)** | Post-hoc | Neural entropy estimator | None | Medium | We use exact entropy, not neural approximation |
| **Single-Pass HalluDet (NeurIPS 2025)** | Generation-time | Single hidden state | None | Very Low | Direct competitor — need to compare F1 scores |
| **ICR Probe / MultiHaluDet** | Generation-time | Hidden-state trajectory | None | Low | We only use single-step states, not trajectories |
| **COIECD / Entropy RAG** | Generation-time | Entropy spikes | None | Very Low | Very close to our Approach A — need differentiation |
| **ACC LLM (Our Project)** | Generation-time | Entropy + Consistency + Hidden-state MLP | None | Low-Medium | Three-signal integration is unique |

### 1.2 Neuroscience-Inspired LLM Architectures

| Method | Brain Region | Implementation | Status |
|--------|-------------|----------------|--------|
| **Webb et al. MAP (Nature Comm. 2025)** | ACC | Explicit module mapping | Published |
| **COCO (arXiv 2026)** | ACC | Conflict neuron identification | Preprint |
| **Rahn et al. EAST (ICLR 2025)** | Prefrontal | Error-aware selective training | Published |
| **Predictive Coding LLMs** | Cortex | Free energy minimization | Preprint |
| **ACC LLM (Our Project)** | ACC | Trained conflict detector + entropy monitor | In progress |

---

## 2. Gap Analysis (To Be Populated)

### 2.1 What Has Been Done (Threats to Novelty)

| Claim We Make | Who Did It | How Similar | Mitigation |
|---------------|-----------|-------------|------------|
| | | | |

### 2.2 What Has NOT Been Done (Opportunities)

| Gap | Why It Matters | How We Fill It |
|-----|---------------|----------------|
| | | |

### 2.3 What We Should Incorporate (Improvements)

| SOTA Technique | Paper | How to Integrate | Effort |
|----------------|-------|------------------|--------|
| | | | |

---

## 3. Novelty Assessment Criteria

### 3.1 Core Claims

1. **"First inference-time verification layer that combines entropy, self-consistency, AND hidden-state classification"**
   - Verify: Is any other paper doing all three simultaneously?
   
2. **"First to apply ACC conflict-monitoring theory as a trained module in LLM generation"**
   - Verify: COCO identifies conflict neurons but doesn't train a detector. MAP maps ACC but doesn't implement hallucination detection.

3. **"Empirical entropy calibration anchored to in-domain distribution"**
   - Verify: Is calibration standard practice? Most papers use fixed thresholds.

4. **"Generation-time hidden-state extraction (not prompt encoding)"**
   - Verify: Single-Pass HalluDet does this too. What's our differentiator?

### 3.2 Experimental Gaps to Fill

- [ ] Compare against Single-Pass HalluDet baseline
- [ ] Compare against COIECD/Entropy RAG baseline
- [ ] Compare against SEPs baseline
- [ ] Ablation: entropy alone vs consistency alone vs detector alone
- [ ] Cross-domain generalization test
- [ ] Human evaluation of flagged vs unflagged outputs

---

*Framework ready for population once all literature sweeps complete.*
