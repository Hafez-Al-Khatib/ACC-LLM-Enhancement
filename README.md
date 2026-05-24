# ACC LLM Enhancement

Fine-tuning Mistral 7B for domain-specific tasks using QLoRA/LoRA on GPU-constrained hardware.

## Hardware Targets

| Platform | VRAM | Strategy |
|----------|------|----------|
| RTX 3080 10GB (Desktop) | 10GB | QLoRA 4-bit + LoRA rank 16-32 |
| Jetson Orin Nano | 8GB unified | QLoRA 4-bit + LoRA rank 8-16 |

## Model

- **Base:** Mistral 7B Instruct v0.3
- **Location:** `models/mistral_7b/` (not tracked in git — download via `setup_model.py`)
- **Quantization:** 4-bit NF4 via bitsandbytes
- **Adapter:** LoRA (r, alpha, dropout configurable)

## Project Structure

```
acc-llm-enhancement/
├── configs/              # Training hyperparameters per experiment
├── src/                  # Core training and inference code
├── scripts/              # Entry points (train.py, eval.py, infer.py)
├── data/                 # Datasets (not tracked in git)
├── adapters/             # Saved LoRA checkpoints
├── notebooks/            # Exploratory analysis
├── models/               # Base model weights (not tracked in git)
├── setup/                # Environment setup scripts
└── requirements.txt      # Python dependencies
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

### 4. Download Model

```bash
python scripts/setup_model.py
```

### 5. Run Training

```bash
python scripts/train.py --config configs/jetson_qlora.yaml
```

## Quick Start — Desktop (RTX 3080)

```bash
# Python 3.12, CUDA 12.1
pip install -r requirements.txt

# Verify
torch: 2.5.1+cu121
bitsandbytes: 0.49.0
peft: 0.19.1

python scripts/train.py --config configs/desktop_qlora.yaml
```

## Training Configs

| Config | Bits | Rank | Alpha | Batch | Accum | LR | Target |
|--------|------|------|-------|-------|-------|-----|--------|
| `jetson_qlora.yaml` | 4 | 8 | 16 | 1 | 8 | 2e-4 | Jetson Orin Nano |
| `desktop_qlora.yaml` | 4 | 32 | 64 | 1 | 4 | 2e-4 | RTX 3080 10GB |

## Memory Budget (Jetson Orin Nano, 8GB)

```
Mistral 7B @ 4-bit NF4:     ~4.0 GB
LoRA adapters (r=8):       ~0.05 GB
Activations / gradients:   ~2.5 GB
Optimizer states (8-bit):  ~0.5 GB
-----------------------------------
Total:                     ~7.1 GB
Headroom:                  ~0.9 GB ✅
```

## WandB / Logging

Optional: Set `WANDB_PROJECT=acc-llm` to track experiments.

## License

Model weights: Apache 2.0 (Mistral). Code: MIT.
