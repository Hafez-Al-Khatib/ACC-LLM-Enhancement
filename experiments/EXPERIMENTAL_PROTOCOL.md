# ACC LLM — Detailed Experimental Protocol

**Version:** 1.0  
**Date:** 2026-05-29  
**Status:** Protocol defined; experiments pending GPU/model availability

---

## 1. Overview

This document specifies the exact experimental procedures for evaluating ACC LLM across four verticals. All experiments follow a standardized pipeline: data preparation → QLoRA fine-tuning → ACC verification layer training → evaluation.

---

## 2. Hardware Configurations

### Configuration A: Desktop (Primary)
- **GPU:** NVIDIA RTX 3080, 10 GB VRAM
- **CPU:** Intel/AMD (sufficient)
- **RAM:** 32 GB recommended
- **OS:** Windows 10/11 or Linux

### Configuration B: Jetson Orin Nano (Edge)
- **GPU:** NVIDIA Ampere (integrated), 8 GB unified memory
- **CPU:** ARM Cortex-A78AE
- **OS:** JetPack 6.0+

---

## 3. Software Environment

### Base Requirements
```bash
# CUDA-enabled PyTorch (critical)
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install transformers==4.46.0 peft==0.14.0 bitsandbytes==0.44.0
pip install datasets==2.21.0 accelerate==0.34.0 trl==0.11.0
pip install sentence-transformers==3.0.0 scikit-learn==1.5.0
pip install wandb  # optional
```

### Environment Verification
```bash
python scripts/check_environment.py
```
Expected output: `READY_STATUS: green` or `yellow` (acceptable).

---

## 4. Phase 1: Model Acquisition

### 4.1 Download Base Model
```bash
# Option A: Authenticated HF Hub (recommended)
huggingface-cli login
python scripts/download_model_robust.py

# Option B: Environment token
HF_TOKEN=hf_xxxxxxxx python scripts/download_model_robust.py
```

**Target files in `models/mistral_7b/`:**
- `model-00001-of-00003.safetensors` (~4.9 GB)
- `model-00002-of-00003.safetensors` (~0.03 GB)
- `model-00003-of-00003.safetensors` (~8.9 GB)
- `model.safetensors.index.json`
- `config.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`

### 4.2 Verify Integrity
```bash
python scripts/validate_model_load.py
```
Expected: `MODEL_LOAD_OK params=7B-ish`

---

## 5. Phase 2: Dataset Preparation

### 5.1 Available Datasets

| Dataset | Vertical | Records | Source | Status |
|---------|----------|---------|--------|--------|
| PubMedQA | Medical | ~1,000 | HuggingFace `pubmed_qa` | Ready |
| SciQ | STEM | ~10,000 | HuggingFace `sciq` | Ready |
| General Instruction | General | ~500 | HuggingFace `fka/awesome-chatgpt-prompts` | Ready |
| FiQA | Financial | TBD | HuggingFace `ChanceFocus/fiqa-sentiment-classification` | Pending |
| Legal | Legal | TBD | HuggingFace `pile-of-law/pile-of-law` | Pending |

### 5.2 Format Specification
All datasets must be converted to JSONL with the following schema:
```json
{
  "instruction": "Question or prompt",
  "input": "Additional context (optional)",
  "output": "Ground-truth answer",
  "vertical": "medical|stem|financial|legal|general"
}
```

### 5.3 Conversion Commands
```bash
python experiments/datasets/auto_load.py
```
This auto-detects and converts available datasets. Check `experiments/datasets/` for outputs.

---

## 6. Phase 3: QLoRA Fine-Tuning

### 6.1 Configuration

**Desktop (`configs/desktop_qlora.yaml`):**
```yaml
model:
  base_model: "models/mistral_7b"
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true

lora:
  r: 32
  lora_alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  output_dir: "adapters/desktop_run"
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 100
  max_seq_length: 1024
  logging_steps: 10
  save_steps: 500
  optim: "paged_adamw_8bit"
  lr_scheduler_type: "cosine"
```

**Jetson (`configs/jetson_qlora.yaml`):**
- `r: 8`, `lora_alpha: 16`
- `target_modules: ["q_proj", "v_proj"]` only
- `max_seq_length: 512`
- `gradient_accumulation_steps: 8`

### 6.2 Training Procedure

```bash
# Single vertical training
python scripts/train.py --config configs/desktop_qlora.yaml --dataset experiments/datasets/pubmedqa/train.jsonl

# Multi-vertical combined training
python scripts/train.py --config configs/desktop_qlora.yaml --dataset experiments/datasets/combined/train.jsonl
```

**Expected runtime:** 2–4 hours for 3 epochs on RTX 3080.

### 6.3 Output Artifacts
- `adapters/desktop_run/` — LoRA adapter checkpoints
- `adapters/desktop_run/final_adapter/` — Final merged adapter

---

## 7. Phase 4: ACC Verification Layer Training

### 7.1 Approach A: Entropy Calibration

```python
from src.acc_layer import EntropyMonitor

monitor = EntropyMonitor(action="flag", window_size=32)
monitor.calibrate(
    model=model,
    tokenizer=tokenizer,
    factual_prompts=calibration_prompts,  # ~100 in-domain factual prompts
    strategy="percentile",  # or "mean_std", "max"
    percentile=95
)
```

**Calibration prompts:** Use the ground-truth answers from the training split of each dataset. Generate ~100 prompts per vertical.

### 7.2 Approach B: Conflict Detector Training

#### Step 1: Generate synthetic conflict data
```bash
python scripts/generate_conflict_data.py \
  --model models/mistral_7b \
  --adapter adapters/desktop_run/final_adapter \
  --output data/acc_training/mistral_conflict_data.jsonl \
  --num_samples_per_class 500 \
  --max_new_tokens 20
```

**Critical:** This MUST be run on Mistral 7B (not tiny GPT-2) to get meaningful 4096-dimensional hidden states.

#### Step 2: Train the MLP classifier
```bash
python scripts/train_conflict_detector.py \
  --data data/acc_training/mistral_conflict_data.jsonl \
  --save_dir adapters/acc_conflict_detector \
  --hidden_ratio 0.5 \
  --dropout 0.1 \
  --epochs 100 \
  --patience 10
```

**Expected output:**
- Validation macro-F1 > 0.60 (acceptable), > 0.75 (good), > 0.85 (excellent)
- Per-class precision/recall for all four classes
- Training history plot (if matplotlib available)

---

## 8. Phase 5: Evaluation

### 8.1 Baseline Conditions

| Condition | Description |
|-----------|-------------|
| Base | Standard generation, no verification |
| ACC-Entropy | Entropy monitoring alone, flag action |
| ACC-SelfConsistency | Multi-candidate consistency (N=5), no entropy |
| ACC-ConflictDetector | Hidden-state MLP alone |
| ACC-Full | Integrated stack: entropy + consistency + detector |

### 8.2 Metrics

#### Primary Metrics
1. **Token-level Hallucination F1**
   - Precision: % of generated tokens present in reference
   - Recall: % of reference tokens generated by model
   - F1: Harmonic mean
   - *Rationale:* Direct measure of factual alignment

2. **Contradiction Rate**
   - Proportion of outputs flagged by self-consistency outlier detection
   - *Rationale:* Measures internal consistency

3. **Calibration Error**
   - ECE (Expected Calibration Error): Bin model outputs by confidence, measure |accuracy - confidence|
   - *Rationale:* Detects overconfident falsehoods

4. **Perplexity (PPL)**
   - exp(-mean log-prob of ground-truth answer)
   - *Rationale:* Measures generative fluency independent of hallucination

#### Secondary Metrics
5. **Latency**
   - Mean generation time per sample (seconds)
   - Breakdown: base generation vs. ACC overhead

6. **Memory Usage**
   - Peak GPU VRAM during inference (GB)

7. **Intervention Rate**
   - % of tokens flagged by entropy monitor
   - % of sequences flagged by conflict detector

### 8.3 Evaluation Commands

```bash
# Run full evaluation suite
python scripts/validate_acc.py \
  --adapter adapters/desktop_run/final_adapter \
  --config configs/desktop_qlora.yaml \
  --dataset experiments/datasets/pubmedqa/test.jsonl \
  --output results/pubmedqa_eval.json

# Run ablation experiments
python experiments/run_ablation.py --config configs/desktop_qlora.yaml
```

### 8.4 Statistical Testing

For each metric, compare conditions using:
- **Wilcoxon signed-rank test** (paired, non-parametric)
- **Significance threshold:** p < 0.05
- **Effect size:** Rank-biserial correlation

Report 95% confidence intervals for all primary metrics.

---

## 9. Phase 6: Ablation Studies

### 9.1 LoRA Rank Ablation
Compare `r=8` (Jetson) vs `r=32` (Desktop) on the same test set.

**Hypothesis:** Higher rank improves task performance but may increase hallucination if the model overfits to training distribution.

### 9.2 Entropy Threshold Strategy Ablation
Compare three calibration strategies on held-out prompts:
1. 95th percentile
2. mean + 2*std
3. Fixed global threshold (H = 2.0 nats)

**Hypothesis:** Domain-calibrated percentile outperforms fixed threshold.

### 9.3 Self-Consistency Depth Ablation
Compare N ∈ {3, 5, 10} candidate continuations.

**Metrics:**
- Contradiction detection rate
- Latency overhead (multiplicative factor)
- Marginal gain per additional candidate

**Hypothesis:** Diminishing returns after N=5.

### 9.4 Domain Generalization Ablation
Train on medical (PubMedQA) only, test on STEM (SciQ) and vice versa.

**Hypothesis:** ACC-Full generalizes better than Base due to domain-agnostic internal signals.

---

## 10. Phase 7: Cross-Hardware Validation

Run the full evaluation on both:
1. Desktop RTX 3080 (10 GB)
2. Jetson Orin Nano (8 GB unified)

**Report:**
- Metric equivalence (within statistical noise)
- Latency ratios
- Memory footprint
- Any hardware-specific failures

---

## 11. Data Management

### 11.1 Version Control
- All code: Git commits with descriptive messages
- Configs: YAML files under version control
- Results: JSON files with timestamp and git hash

### 11.2 Reproducibility Checklist
- [ ] Seed all random number generators
- [ ] Pin all dependency versions
- [ ] Record CUDA/driver versions
- [ ] Save exact training commands
- [ ] Archive model checkpoints with config

### 11.3 Seed Configuration
```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## 12. Expected Timeline

| Phase | Task | Estimated Time | Dependencies |
|-------|------|---------------|--------------|
| 1 | Model download | 30–60 min | HF auth, internet |
| 2 | Dataset prep | 15 min | auto_load.py |
| 3 | QLoRA training (per vertical) | 2–4 hours | GPU, model |
| 4 | ACC layer training | 30–60 min | Trained adapter |
| 5 | Evaluation (per vertical) | 1–2 hours | All above |
| 6 | Ablations | 4–8 hours | Base results |
| 7 | Cross-hardware | 2–4 hours | Jetson access |
| — | **Total (single vertical)** | **~8–16 hours** | — |
| — | **Total (all verticals + ablations)** | **~40–60 hours** | — |

---

## 13. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUDA OOM during training | Medium | Reduce batch size, seq length, or LoRA rank |
| HF Hub rate limit | High | Use `HF_TOKEN`, cache model locally |
| Poor conflict detector F1 | Medium | Increase synthetic data size, tune architecture |
| Dataset 404/missing | Medium | Use fallback datasets, document exclusions |
| Jetson OOM | High | Reduce to q/v LoRA only, seq_len=512 |

---

## 14. Success Criteria

### Minimum Viable Result
- QLoRA training completes without OOM
- Conflict detector achieves macro-F1 > 0.50 on validation
- ACC-Full shows measurable improvement over Base on at least one metric

### Good Result
- Conflict detector macro-F1 > 0.70
- ACC-Full significantly outperforms Base (p < 0.05) on hallucination F1
- Self-consistency detects >10% contradictions in test set

### Excellent Result
- Conflict detector macro-F1 > 0.80
- ACC-Full outperforms all ablated conditions
- Cross-hardware metrics are equivalent
- Paper-ready figures and tables generated

---

*End of Protocol*
