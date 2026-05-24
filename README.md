# ACC LLM Enhancement

Fine-tuning Mistral 7B for domain-specific tasks using QLoRA/LoRA on GPU-constrained hardware.
Supports systematic multi-vertical ablation studies across Medical, Legal, Financial, STEM, and General instruction-following domains.

## Hardware Targets

| Platform | VRAM | Strategy |
|----------|------|----------|
| RTX 3080 10GB (Desktop) | 10GB | QLoRA 4-bit + LoRA rank 16-32 |
| Jetson Orin Nano | 8GB unified | QLoRA 4-bit + LoRA rank 4-8 |

## Model

- **Base:** Mistral 7B Instruct v0.3
- **Location:** `models/mistral_7b/` (not tracked in git — download via `scripts/setup_model.py`)
- **Quantization:** 4-bit NF4 via bitsandbytes
- **Adapter:** LoRA (r, alpha, dropout configurable per experiment)

## Project Structure

```
acc-llm-enhancement/
├── configs/              # Training hyperparameters per platform
│   ├── desktop_qlora.yaml
│   └── jetson_qlora.yaml
├── src/                  # Core training and inference code
│   └── model_utils.py    # Model loading, LoRA, BNB config
├── scripts/              # Entry points
│   ├── train.py          # Main training (TRL SFTTrainer)
│   ├── infer.py          # Inference with adapter
│   └── setup_model.py    # Download model from HuggingFace
├── experiments/          # Multi-vertical ablation framework
│   ├── registry.json     # Experiment definitions by vertical
│   ├── datasets/         # Auto-download + format from HF hub
│   │   └── auto_load.py  # PubMedQA, FiQA, SciQ, Alpaca, etc.
│   ├── benchmarks/       # Evaluation scripts per vertical
│   ├── results/          # JSONL result logs
│   ├── templates/        # Reusable config templates
│   ├── run_ablation.py   # Batch experiment orchestrator
│   └── summarize.py      # Results aggregation
├── data/                 # Local datasets (not tracked in git)
├── adapters/             # Saved LoRA checkpoints (not tracked)
├── notebooks/            # Analysis notebooks
├── models/               # Base model weights (not tracked)
├── setup/                # Environment setup scripts
│   └── JETSON_NOTES.md
└── requirements.txt
```

## Quick Start — Ablation Study

### 1. Download Model

```bash
python scripts/setup_model.py
```

### 2. Run a Single Vertical

```bash
# Auto-download PubMedQA, format, and train
python experiments/datasets/auto_load.py \
    --dataset pubmedqa \
    --output-dir experiments/datasets/pubmedqa

python scripts/train.py --config configs/desktop_qlora.yaml
# (update config dataset paths to point to experiments/datasets/pubmedqa)
```

### 3. Run Systematic Ablations

```bash
python experiments/run_ablation.py \
    --vertical medical \
    --dataset pubmedqa \
    --ablate rank \
    --values 4 8 16 32 \
    --hardware desktop \
    --config-template configs/desktop_qlora.yaml \
    --epochs 3
```

This generates one config per rank, trains sequentially, and logs all results to `experiments/results/`.

### 4. Summarize Results

```bash
python experiments/summarize.py --results experiments/results/medical_pubmedqa_rank_2026*.jsonl
```

## Quick Start — Jetson Orin Nano

Jetson runs **ARM64 (aarch64)**. Standard pip wheels won't work — use NVIDIA's Jetson-specific builds.

### 1. Install Jetson PyTorch

```bash
# JetPack 6.x (CUDA 12.2)
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.4.0-cp310-cp310-linux_aarch64.whl

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 2. Install bitsandbytes (ARM64 Build)

```bash
# Pre-built wheel for Jetson (community)
pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.7.1/bitsandbytes-0.43.1-cp310-cp310-linux_aarch64.whl

# If above fails, build from source:
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=cuda -D CMAKE_CUDA_ARCHITECTURES=87 ..
make
pip install .
```

### 3. Install Remaining Dependencies

```bash
pip install transformers accelerate peft trl datasets huggingface-hub
```

### 4. Clone This Repo on Jetson

```bash
git clone https://github.com/Hafez-Al-Khatib/ACC-LLM-Enhancement.git
cd ACC-LLM-Enhancement
python scripts/setup_model.py
python scripts/train.py --config configs/jetson_qlora.yaml
```

See `setup/JETSON_NOTES.md` for full details.

## Verticals & Datasets

| Vertical | Datasets | Status |
|----------|----------|--------|
| Medical | PubMedQA, MedMCQA, MedAlpaca | Planned |
| Legal | Pile of Law, Legal QA | Planned |
| Financial | FiQA, Financial PhraseBank | Planned |
| STEM | SciQ, OpenBookQA, ScienceQA | Planned |
| General | Alpaca, Dolly, OpenAssistant | Planned |

All datasets auto-download via `experiments/datasets/auto_load.py`.

## Training Configs

| Config | Bits | Rank | Alpha | Batch | Accum | LR | Seq | Target |
|--------|------|------|-------|-------|-------|-----|------|--------|
| `desktop_qlora.yaml` | 4 | 32 | 64 | 1 | 4 | 2e-4 | 1024 | RTX 3080 |
| `jetson_qlora.yaml` | 4 | 8 | 16 | 1 | 8 | 2e-4 | 512 | Orin Nano |

## Memory Budget (Jetson Orin Nano, 8GB)

```
Mistral 7B @ 4-bit NF4:     ~4.0 GB
LoRA adapters (r=8):        ~0.05 GB
Activations / gradients:      ~2.5 GB
Optimizer states (8-bit):   ~0.5 GB
-----------------------------------
Total:                      ~7.1 GB
Headroom:                   ~0.9 GB ✅
```

## WandB / Logging

Optional: Set `WANDB_PROJECT=acc-llm` to track experiments.
Each ablation run auto-generates a unique run name: `{vertical}_{dataset}_{param}{value}_{hardware}_{timestamp}`.

## Citation

If you use this framework, cite:
- Mistral 7B: Jiang et al., 2023
- QLoRA: Dettmers et al., 2023
- TRL: von Werra et al., 2020

## License

Model weights: Apache 2.0 (Mistral). Code: MIT.
