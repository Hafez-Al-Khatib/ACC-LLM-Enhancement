# Jetson Orin Nano — Setup Notes

## Architecture
- ARM64 (aarch64) — standard x86_64 wheels will NOT install
- 8GB unified memory (CPU + GPU shared)
- Ampere GPU, 32 Tensor Cores, Compute Capability 8.7
- JetPack 6.x ships CUDA 12.2

## PyTorch for Jetson

### Option A: NVIDIA Pre-built Wheel (Recommended)
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.4.0-cp310-cp310-linux_aarch64.whl
```

### Option B: Build from Source (if you need a specific version)
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.4.0
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'/usr'}"
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py install
```

## bitsandbytes for ARM64

### Option A: Pre-built (check community repos)
```bash
# Check if a recent pre-built exists for cc8.7
pip install https://github.com/TimDettmers/bitsandbytes/releases/download/0.43.3/bitsandbytes-0.43.3-cp310-cp310-linux_aarch64.whl
```

### Option B: Build from Source
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=87 -S .
cmake --build . --config Release
pip install .
```

### Option C: Use GPTQ / AWQ instead of QLoRA
If bitsandbytes refuses to build on Jetson, switch quantization strategy:
- **AutoGPTQ** — has ARM64 wheels
- **AutoAWQ** — may need build
- **llama.cpp** — CPU/GPU hybrid inference (no training though)

## Flash Attention

Flash Attention v2 is NOT available for Jetson Ampere (only Hopper/Ada).
Use `transformers` default SDPA attention — it's fine for 512-token sequences.

## Memory Budget Check

```bash
# After setup, verify with:
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

Expected: ~7.6 GB usable on Jetson Orin Nano 8GB.

## Dataset for Jetson

Keep datasets small. Jetson storage is typically eMMC or SD card:
- Use JSONL, not Parquet (lighter)
- Target <10K samples for first experiments
- Use `packing=True` in SFTTrainer to maximize throughput

## SSH / Remote Development

Recommended workflow:
1. Code on desktop (VS Code + Remote-SSH extension)
2. Push to GitHub
3. Pull on Jetson via SSH
4. Run training in `tmux` or `screen` session

```bash
# On Jetson, start detached training
screen -S llm-train
python scripts/train.py --config configs/jetson_qlora.yaml
# Ctrl+A, D to detach
```

## Common Issues

| Issue | Fix |
|-------|-----|
| `Illegal instruction` | Wrong PyTorch wheel — must be aarch64 |
| `CUDA out of memory` | Reduce `max_seq_length` to 256, use `r=4` |
| `bitsandbytes` build fails | Use `CMAKE_CUDA_ARCHITECTURES=87`, ensure CUDA 12.2 |
| Slow training | Expected — Jetson Ampere is ~10× slower than RTX 3080. Use smaller datasets. |

## Suggested First Run

```bash
git clone https://github.com/YOUR_USERNAME/acc-llm-enhancement.git
cd acc-llm-enhancement

# Download model (~15GB download on Jetson — use Ethernet)
python scripts/setup_model.py --model-id mistralai/Mistral-7B-Instruct-v0.3

# Create tiny synthetic dataset for smoke test
cat > data/train.jsonl << 'EOF'
{"text": "### Instruction:\nSummarize the following.\n\n### Input:\nKidney fibrosis is characterized by...\n\n### Response:\nProgressive scarring of kidney tissue."}
{"text": "### Instruction:\nWhat is a drug target?\n\n### Input:\n\n\n### Response:\nA biological molecule that a drug acts upon."}
EOF

# Run training
python scripts/train.py --config configs/jetson_qlora.yaml
```
