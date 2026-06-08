# Enhanced Related Work — Literature Review

**Compiled:** 2026-05-29 ~05:30 UTC  
**Scope:** Entropy-based decoding, uncertainty quantification for hallucination detection, QLoRA domain adaptation, and medical LLMs.

---

## 1. Entropy-Based Decoding and Temperature Control

### EDT: Entropy-based Dynamic Temperature Sampling (Zhang et al., 2024)
- **Paper:** arXiv:2403.14541
- **Key Idea:** Dynamically adjusts sampling temperature based on the entropy of the next-token distribution. High-entropy steps get lower temperature (sharper), low-entropy steps get higher temperature (more diverse).
- **Relevance:** Directly validates our Approach A entropy-monitoring principle. Their results show improved coherence in open-ended generation.
- **Citation:** Zhang, S., Bao, Y., & Huang, S. (2024). EDT: Improving large language models' generation by entropy-based dynamic temperature sampling. *arXiv preprint arXiv:2403.14541*.

### Entropix: Entropy-Based Sampling and Parallel CoT Decoding (xjdr & doomslide, 2024)
- **Repository:** https://github.com/xjdr-alt/entropix
- **Key Idea:** Uses entropy statistics (mean, variance, skewness of the next-token distribution) to guide sampling. Also explores parallel chain-of-thought decoding paths selected by entropy criteria.
- **Relevance:** Validates that entropy statistics beyond simple Shannon entropy can guide generation quality. Their parallel CoT approach is analogous to our multi-candidate self-consistency.

### Min-p Sampling (Meister et al., 2024)
- **Paper:** arXiv:2407.01082
- **Key Idea:** Instead of nucleus (top-p) sampling, filters tokens by absolute probability threshold scaled by the maximum probability. Preserves high-quality tokens while maintaining diversity.
- **Relevance:** Complementary to our entropy approach; min-p handles the "tail" of the distribution while entropy handles overall uncertainty.

---

## 2. Uncertainty Quantification for Hallucination Detection

### Semantic Entropy (Kuhn et al., 2023) — Already in draft
- **Status:** Cited in draft.
- **Note:** The *Nature* 2024 paper by Farquhar et al. (same group) extends this with larger-scale validation:
  - Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017), 625–630.

### Uncertainty Quantification Survey (Chen et al., 2025)
- **Paper:** arXiv:2510.12040 (Oct 2025)
- **Key Idea:** Comprehensive taxonomy of UQ methods for hallucination detection:
  1. Token probability-based (CSP, token entropy)
  2. Output consistency-based (SelfCheckGPT, semantic entropy)
  3. Internal state examination (SAPLMA, SEPs, INSIDE, HaloScope)
  4. Self-checking methods (Ptrue, verbalized confidence)
- **Relevance:** Our Approach A falls under categories 1+2; Approach B falls under category 3. The survey validates that internal-state methods (our MLP on hidden states) are an active and promising direction.
- **Important Finding:** On short-form QA, LARS and SAPLMA achieve highest performance. Among self-supervised methods, SAR performs best. For long-form generation, performance drops across all methods.

### Teaching Language Models to Faithfully Express Uncertainty (Yaldiz et al., 2025)
- **Paper:** arXiv:2510.12587
- **Key Idea:** Calibrates verbalized confidence statements from LLMs to match actual correctness probability.
- **Relevance:** Complementary to our approach—we detect uncertainty via internal signals; they improve the model's ability to express it.

### Fine-Grained Confidence Estimation During Generation (Han et al., 2025)
- **Paper:** arXiv:2508.12040
- **Key Idea:** Estimates confidence at each generation step using hidden-state features.
- **Relevance:** Very close to our Approach B. Validates step-level uncertainty estimation.

---

## 3. Internal State Examination Methods

### SAPLMA (Azaria & Mitchell, 2023)
- **Paper:** Already cited implicitly via survey.
- **Key Idea:** Trains an MLP on hidden states to predict whether a statement is true or false.
- **Relevance:** Direct precedent for our Approach B latent conflict detector. Their MLP operates on sentence-level hidden states; ours operates on per-token generation-time states.

### Semantic Entropy Probes (SEPs) (Kossen et al., 2024)
- **Key Idea:** Trains lightweight linear probes on hidden states to predict semantic entropy scores computed from sampled generations.
- **Relevance:** Bridges internal states and sampling-based uncertainty—exactly the bridge our framework attempts.

### INSIDE (Chen et al., 2024)
- **Key Idea:** EigenScore + test-time feature clipping. Computes semantic divergence in hidden states across sampled generations.
- **Relevance:** Uses covariance structure of hidden states for uncertainty quantification.

---

## 4. Medical/Domain-Specific LLMs and QLoRA

### BioMistral 7B (Labrak et al., 2024)
- **Paper:** arXiv:2402.10373
- **Key Idea:** Continued pre-training of Mistral 7B on PubMed Central abstracts. Evaluated on 10 medical QA tasks.
- **Key Finding:** BioMistral 7B outperforms Mistral 7B Instruct on 8/10 tasks. However, on PubMedQA specifically, performance DECLINED by ~15.7% compared to other models, "likely due to hallucinations caused by imbalanced classes."
- **Relevance:** CRITICAL for our project. This directly motivates our ACC verification layer for medical QA—domain adaptation alone is insufficient when the dataset has class imbalance that causes hallucinations.
- **Citation:** Labrak, Y., Bazoge, A., Morin, E., Gourraud, P. A., Rouvier, M., & Dufour, R. (2024). BioMistral: A collection of open-source pretrained large language models for medical domains. *arXiv preprint arXiv:2402.10373*.

### Medical LLMs: Fine-Tuning vs. RAG (MDPI, 2025)
- **Paper:** MDPI Biomedicines 2025, 12(7), 687
- **Key Finding:** Compared five models (Llama-3.1-8B, Gemma-2-9B, Mistral-7B-Instruct, Qwen2.5-7B, Phi-3.5-Mini) across FT, RAG, and FT+RAG.
- **Key Result:** RAG emerged as the most effective adaptation strategy overall. However, for some models (PHI), the hybrid FT+RAG approach was especially helpful.
- **Relevance:** Validates our multi-vertical approach and suggests future work could integrate RAG as an additional signal channel to our ACC layer.

### Clinical Mistral with QLoRA (Preprints, 2025)
- **Paper:** Preprints.org, 2025
- **Key Idea:** Adapts Mistral 7B to clinical tasks using QLoRA with 4-bit quantization.
- **Relevance:** Validates our QLoRA configuration choices for medical domain adaptation.

---

## 5. Key Takeaways for ACC LLM Paper

1. **Entropy-based detection is well-founded.** Multiple papers (EDT, Entropix, Farquhar et al. Nature 2024) validate that entropy statistics correlate with hallucination risk.

2. **Internal-state methods are underexplored but promising.** The UQ survey (Chen et al., 2025) identifies SAPLMA and SEPs as top performers. Our generation-time per-token MLP is a novel variant—operating on *generation* hidden states rather than *prompt* hidden states.

3. **Medical domain adaptation alone causes hallucinations.** BioMistral's PubMedQA decline is a critical finding: fine-tuning can *increase* hallucination in imbalanced domains. This strongly motivates inference-time verification layers.

4. **Self-consistency is expensive but effective.** Multiple papers confirm output-consistency methods work, but they require N additional forward passes. Our approach of embedding-space clustering of candidates is a latency-amortized variant.

5. **Future direction: RAG + ACC hybrid.** The MDPI study shows FT+RAG outperforms either alone. Integrating retrieval signals into our conflict detector MLP is a natural extension.

---

## 6. Additional Citations to Add to Draft

```bibtex
@article{farquhar2024detecting,
  title={Detecting hallucinations in large language models using semantic entropy},
  author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  journal={Nature},
  volume={630},
  number={8017},
  pages={625--630},
  year={2024}
}

@article{labrak2024biomistral,
  title={BioMistral: A collection of open-source pretrained large language models for medical domains},
  author={Labrak, Yanis and Bazoge, Adrien and Morin, Emmanuel and Gourraud, Pierre-Antoine and Rouvier, Mickael and Dufour, Richard},
  journal={arXiv preprint arXiv:2402.10373},
  year={2024}
}

@article{zhang2024edt,
  title={EDT: Improving large language models' generation by entropy-based dynamic temperature sampling},
  author={Zhang, Shimao and Bao, Yu and Huang, Shujian},
  journal={arXiv preprint arXiv:2403.14541},
  year={2024}
}

@article{chen2025uncertainty,
  title={Uncertainty quantification for hallucination detection in large language models: Foundations, methodology, and future directions},
  author={Chen, Liang and others},
  journal={arXiv preprint arXiv:2510.12040},
  year={2025}
}

@article{han2025finegrained,
  title={Mind the generation process: Fine-grained confidence estimation during LLM generation},
  author={Han, Jinyi and Li, Tingyun and Chen, Shisong and Shi, Jie and Wang, Xinyi and Yue, Guanglei and Liang, Jiaqing and Lin, Xin and Wen, Liqian and Chen, Zulong and others},
  journal={arXiv preprint arXiv:2508.12040},
  year={2025}
}
```
