# ACC LLM Enhancement

Fine-tuning Mistral 7B for domain-specific tasks using QLoRA/LoRA on GPU-constrained hardware.
Supports systematic multi-vertical ablation studies across Medical, Legal, Financial, STEM, and General instruction-following domains.

## Hardware Targets

| Platform | VRAM / Memory | Strategy |
|----------|--------------|----------|
| RTX 3080 10GB (Desktop) | 10GB | QLoRA 4-bit + LoRA rank 16-32 |
| Intel Arc 140T (Zenbook Duo) | 16GB unified | FP16 or QLoRA 4-bit |
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

## Current Status (Overnight Update)

**Last updated:** 2026-05-29 ~05:40 UTC

### What's Ready
- **Code fixed:** ~10 critical bugs patched (paths, WandB, entropy, padding, etc.)
- **Pipeline validated:** Full train → validate flow tested on `sshleifer/tiny-gpt2`
- **Datasets ready:** PubMedQA (medical), SciQ (STEM), General Instruction
- **Documentation:** Detailed experimental protocol, enhanced related work, BibTeX references
- **Utility scripts:** Environment checker, auto-launcher, results aggregator

### What's Blocked
- **Mistral 7B download:** Requires HuggingFace authentication (`huggingface-cli login`)

### Next Steps (See `results/WAKE_UP_GUIDE.md`)
1. Authenticate HF Hub and download model
2. Install the correct PyTorch backend for your hardware (see Environment Setup below)
3. Launch QLoRA training per `experiments/EXPERIMENTAL_PROTOCOL.md`

---

## Environment Setup

The codebase supports **three hardware backends**. Pick the one matching your machine and install into a virtual environment.

### Intel Arc / XPU (e.g. ASUS Zenbook Duo, Core Ultra with Arc 140T)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements-xpu.txt
```

This installs PyTorch 2.8.0 with **native Intel XPU support** (no IPEX needed). The wheel is ~1.2 GB, so the first install may take 20–40 minutes on a slow connection.

Verify the GPU is visible:

```python
import torch
print(torch.xpu.is_available())        # True
print(torch.xpu.get_device_name(0))    # Intel(R) Arc(TM) ...
```

### NVIDIA CUDA (e.g. RTX 3080, RTX 4090, A100)

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements-cuda.txt
```

Verify:

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX ...
```

### CPU-Only (not recommended for 7B training)

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements-cpu.txt
```

---

## Loading Models on Your Platform

`src.model_utils.load_model` auto-detects the best device, or you can specify it explicitly:

```python
from src.model_utils import load_model, load_tokenizer, make_bnb_config

# Intel Arc — loads to CPU first, then moves to XPU automatically
model = load_model("models/mistral_7b", device="xpu")

# NVIDIA — standard device_map
model = load_model("models/mistral_7b", device="cuda:0")

# With 4-bit QLoRA on any platform
bnb = make_bnb_config({"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"})
model = load_model("models/mistral_7b", bnb_config=bnb, device="xpu")
```

---

## Generation-Time Conflict Detection

The framework now supports **real-time conflict detection** during generation via a
`PredictiveCodingDetector` integrated into the HuggingFace `.generate()` pipeline.
This enables per-token intervention (flag, regenerate, suppress, warning) based on
a fusion of entropy monitoring and hierarchical prediction-error signals.

### Quick Example

```python
from src.acc_integration import ACCEnhancedGenerator, MarkerConfig, UnifiedDecisionEngine
from src.acc_conflict_detector import PredictiveCodingDetector

# Load a pre-trained conflict detector
detector = PredictiveCodingDetector(hidden_dim=4096)
detector.load_state_dict(torch.load("detector.pt"))

gen = ACCEnhancedGenerator(
    model=model,
    tokenizer=tokenizer,
    action="flag",
    use_conflict_detector=True,
    use_realtime_conflict_detector=True,
    conflict_detector=detector,
    decision_engine=UnifiedDecisionEngine(
        entropy_threshold=1.5,
        conflict_score_threshold=0.7,
        dual_signal_regenerate=True,
        marker_config=MarkerConfig(
            hallucination=" [HALLUCINATION]",
            contradiction=" [CONTRADICTION]",
        ),
    ),
)

output = gen.generate_from_prompt(
    "What is the capital of France?",
    max_new_tokens=50,
    return_dict_in_generate=True,
)

print(output.text[0])          # generated text with inline markers
print(gen.explain_decisions(output))  # human-readable decision trace
```

### Output Format

`ACCGenerationOutput` includes:

| Field | Description |
|-------|-------------|
| `text` | Generated text with intervention markers |
| `per_token_entropy` | Shannon entropy at each step |
| `per_token_decisions` | List of `{action, reason, entropy, conflict_score, primary, secondary}` |
| `regenerations` | Count of regeneration interventions per sequence |
| `primary_labels` | 3-way label per token (`supported`/`unsupported`/`uncertain`) |
| `secondary_labels` | 2-way label per token (`hallucinated`/`contradictory`) |
| `conflict_scores` | Continuous conflict score [0, 1] per token |

### Intervention Actions

| Action | Trigger | Effect |
|--------|---------|--------|
| `pass` | No signal | None |
| `flag` | Entropy breach or moderate conflict | Insert marker after token |
| `warning` | Same as flag (configurable) | Insert prefix before token |
| `regenerate` | Dual signal (entropy + conflict) | Resample with lower temperature |
| `suppress` | High conflict on unsupported token | Block top-1 token, force alternative |

---

## Quick Start — Ablation Study

### 1. Verify Environment

```bash
python scripts/check_environment.py
```

### 2. Download Model (Authenticated)

```bash
huggingface-cli login
python scripts/download_model_robust.py
```

### 3. Run a Single Vertical

```bash
# Auto-download and format datasets
python experiments/datasets/auto_load.py

# Train
python scripts/train.py --config configs/desktop_qlora.yaml
```

### 4. Run Systematic Ablations

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

### 5. Summarize Results

```bash
python experiments/summarize.py --results experiments/results/*.jsonl
python scripts/aggregate_results.py --input results/ --output results/summary/
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
