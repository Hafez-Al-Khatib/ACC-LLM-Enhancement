# Literature Sweep: Efficient Fine-Tuning & Edge Deployment

**Date:** 2026-05-30  
**Scope:** 2024–2025 research on QLoRA, DoRA, quantization, speculative decoding, and edge deployment

---

## Key Findings

### Advanced PEFT Methods

| Method | Venue | Year | Key Innovation |
|--------|-------|------|---------------|
| **DoRA** | ICML | 2024 | Weight-Decomposed Low-Rank Adaptation; better stability than LoRA |
| **PiSSA** | NeurIPS | 2024 | Principal Singular values and Singular vectors Adaptation; superior initialization |
| **TIES** | NeurIPS | 2023 | Task-specific adapter merging with conflict resolution |
| **DARE / DARE-TIES** | ICML | 2024 | Drop And REscale for merging adapters without interference |

### Quantization Advances

| Method | Venue | Year | Key Innovation |
|--------|-------|------|---------------|
| **AWQ** | MLSys | 2024 (Best Paper) | Activation-aware weight quantization; better than NF4 |
| **AQLM** | ICML | 2024 | Accurate quantization with learned codebooks |
| **GPTQ** | — | 2023 | Post-training quantization standard |

### Edge Deployment

| Method | Venue | Year | Key Innovation |
|--------|-------|------|---------------|
| **EdgeLLM** | IEEE TMC | 2024/2025 | Speculative decoding on smartphones/Jetson; 2.9–9.3× speedup |
| **PowerInfer-2** | — | 2024 | Sparsity-aware scheduling; 11.68 tok/sec with Mixtral-47B on smartphones |
| **Confidant** | MobiCom | 2025 | Pipeline-parallel fine-tuning across mobile devices |

### Multi-Stage Training Recipes

| Method | Venue | Year | Key Innovation |
|--------|-------|------|---------------|
| **SaulLM** | NeurIPS | 2024 | CPT → SFT → Alignment pipeline for legal domain |
| **Financial LLMs** | EMNLP | 2025 | Multi-stage domain adaptation for finance |

---

## Actionable Recommendations

| Recommendation | Priority | Supporting Papers |
|----------------|----------|-------------------|
| **Replace vanilla LoRA with DoRA + PiSSA initialization** | High | DoRA (ICML 2024), PiSSA (NeurIPS 2024) |
| **Adopt AWQ for 4-bit quantization** on Jetson | High | AWQ (MLSys 2024 Best Paper) |
| **Design hybrid RAG + QLoRA pipeline** | High | Agriculture RAG/FT study (2024) |
| **Use TIES or DARE-TIES** for adapter merging | Medium | TIES (NeurIPS 2023), DARE (ICML 2024) |
| **Evaluate speculative decoding** for Jetson | Medium | EdgeLLM (IEEE TMC 2024/2025) |
| **Follow multi-stage recipe** (CPT → SFT → Alignment) | Medium | SaulLM (NeurIPS 2024) |

---

*Report compiled by subagent literature scout.*
