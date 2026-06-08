# 🌅 ACC LLM — Overnight Wake-Up Guide

**Compiled:** 2026-05-29 ~04:15 UTC  
**Your status:** Asleep (dawn departure)  
**Agent status:** Autonomous overnight work completed

---

## TL;DR — Read This First

1. ✅ **Code is fixed and ready.** Multiple critical bugs patched (see below).
2. ⚠️ **Mistral 7B download FAILED.** Unauthenticated HF Hub is severely rate-limited (0 bytes transferred after 10+ minutes). You MUST authenticate or use an alternative method:
   ```bash
   huggingface-cli login
   # OR set HF_TOKEN environment variable
   HF_TOKEN=your_token python scripts/download_model_robust.py
   ```
3. ⚠️ **THIS ENVIRONMENT HAS NO GPU.** PyTorch is CPU-only (`2.12.0+cpu`). You cannot train Mistral here. You must either:
   - Install CUDA PyTorch on this machine (if it has your RTX 3080), OR
   - Transfer `models/mistral_7b/` to your actual GPU workstation.
4. ✅ **Tiny GPT-2 smoke test passed.** Full pipeline (train → validate ACC) works end-to-end.

---

## 1. Overnight Accomplishments

### Code Fixes (Critical)
| Fix | File | Details |
|-----|------|---------|
| Typo in deps | `requirements.txt` | `transforms` → `transformers` |
| Missing base model path | `configs/acc_test.yaml` | Was pointing to non-existent local path; now `sshleifer/tiny-gpt2` |
| Hardcoded paths | Multiple scripts | Removed all `D:/ACC LLM Enhancement/` paths |
| `local_files_only=True` footgun | `infer.py`, `validate_*.py` | Now falls back to HF Hub when local files missing |
| WandB crashes training | `scripts/train.py` | Wrapped `wandb.init()` in try/except; added `report_to=[]` |
| Entropy regeneration bug | `src/acc_layer.py` | Changed `/ multiplier` (raises temp) → `* multiplier` (lowers temp) |
| Padding in loss | `scripts/train.py` | Labels now mask padding tokens with `-100` so pad tokens don't train |
| Layer clamping | `scripts/generate_conflict_data.py` | Auto-clamps layer index for models with < 4 layers |
| Ground-truth preservation | `experiments/datasets/auto_load.py` | Preserves `instruction/input/output` fields for evaluation |

### Infrastructure Created
- `results/overnight_status_report.md` — Full status report
- `scripts/auto_launch_training.py` — Watches for model shards and auto-launches training
- `results/acc_validation_tiny_gpt2_general.log` — Smoke test results

### Datasets Ready
| Dataset | Vertical | Path |
|---------|----------|------|
| PubMedQA | Medical | `experiments/datasets/pubmedqa/` |
| SciQ | STEM | `experiments/datasets/sciq/` |
| General Instruction | General | `experiments/datasets/general_instruction/` |

**Missing:** Alpaca (HF timeout), FiQA (dataset ID 404), Financial PhraseBank (extremely slow download). These can be added later; core experiments can start with PubMedQA + SciQ.

---

## 2. Mistral Download Status

```
Location:     models/mistral_7b/
Method:       hf_hub_download (HF Hub)
Status:       FAILED — requires authentication
Cache size:   ~3.3 MB (small files only)
Target:       ~15 GB (3 shards + tokenizer + config)
ETA:          Depends on connection speed after auth
```

**Problem:** Unauthenticated HF Hub downloads are severely rate-limited. After 10+ minutes, 0 bytes of model shards were transferred.

**Solution:** Log in with your HuggingFace token:
```bash
huggingface-cli login
# Then run:
python scripts/download_model_robust.py
```

Or use an environment variable:
```bash
export HF_TOKEN=your_token_here
python scripts/download_model_robust.py
```

When complete, the following files should appear in `models/mistral_7b/`:
```
model-00001-of-00003.safetensors  (~4.9 GB)
model-00002-of-00003.safetensors  (~0.03 GB)
model-00003-of-00003.safetensors  (~8.9 GB)
model.safetensors.index.json
tokenizer.model, tokenizer.json, config.json, etc.
```

---

## 3. The GPU Problem ⚠️

**Current PyTorch:** `2.12.0+cpu` (CPU-only)  
**Required for training:** CUDA-enabled PyTorch

### Check if you have an NVIDIA GPU on this machine
```bash
nvidia-smi
```

If `nvidia-smi` is not found, you may need to:
1. Install NVIDIA drivers
2. Reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If this machine does NOT have a GPU, you must transfer the downloaded model to your RTX 3080 machine:
```bash
# On this machine (compress)
tar czf mistral_7b.tar.gz models/mistral_7b/

# Transfer to GPU machine, then extract
tar xzf mistral_7b.tar.gz
```

---

## 4. Quick-Start Checklist (When You're Ready)

### Step 1: Verify environment
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Should print: True 12.1 (or similar)
```

### Step 2: Verify model download
```bash
python scripts/validate_model_load.py
# Expected: MODEL_LOAD_OK params=7,000,000,000-ish
```

### Step 3: Launch QLoRA training
```bash
python scripts/train.py --config configs/desktop_qlora.yaml
```
Expected runtime: ~2–4 hours for 3 epochs on RTX 3080 10GB.

### Step 4: Generate conflict detector training data
```bash
python scripts/generate_conflict_data.py --model models/mistral_7b --output data/acc_training/mistral_conflict_data.jsonl
```
This will create ~2,000 labeled hidden-state records with **4096-dimensional** vectors (unlike the useless 2D tiny-GPT2 vectors).

### Step 5: Train conflict detector
```bash
python scripts/train_conflict_detector.py --data data/acc_training/mistral_conflict_data.jsonl --save_dir adapters/acc_conflict_detector
```

### Step 6: Run full ACC validation
```bash
python scripts/validate_acc.py --adapter adapters/desktop_run/final_adapter --config configs/desktop_qlora.yaml --device cuda
```

### Step 7: Run ablation experiments
```bash
python experiments/run_ablation.py --config configs/desktop_qlora.yaml
```

---

## 5. Known Issues & Limitations

1. **Tiny GPT-2 is NOT representative.** The smoke test confirmed the pipeline runs, but:
   - Entropy is always ~10.8 nats (near-random model)
   - Hidden states are 2D (useless for conflict detector)
   - Self-consistency always scores 1.0 (all outputs are similar gibberish)
   **Real results require Mistral 7B.**

2. **Missing datasets.** Financial (FiQA, Financial PhraseBank) and legal verticals lack datasets. Recommend:
   - `ChanceFocus/fiqa-sentiment-classification` for financial sentiment
   - `pile-of-law/pile-of-law` or `lex_glue` for legal

3. **HF Hub rate limits.** Unauthenticated downloads are slow. Consider:
   ```bash
   huggingface-cli login
   ```

4. **WandB is optional.** All training runs work without it. If you want WandB:
   ```bash
   wandb login
   ```

---

## 6. If Something Is Broken

**Training crashes with CUDA OOM:**
- Reduce `max_seq_length` to 512 in `configs/desktop_qlora.yaml`
- Reduce `per_device_train_batch_size` to 1 (already set)
- Increase `gradient_accumulation_steps` to 8

**Model load fails after download:**
```bash
python scripts/validate_model_load.py
```
Check the error. Most common: missing tokenizer files (re-run download).

**Conflict detector training fails:**
- Ensure you ran `generate_conflict_data.py` on Mistral 7B first
- Check that the JSONL file has non-empty `hidden_state` arrays

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `results/overnight_status_report.md` | Full overnight status |
| `results/acc_validation_tiny_gpt2_general.log` | Smoke test output |
| `scripts/auto_launch_training.py` | Auto-launcher (waits for shards, trains) |
| `scripts/validate_model_load.py` | Quick model integrity check |
| `scripts/train.py` | QLoRA training |
| `scripts/generate_conflict_data.py` | Synthetic conflict data |
| `scripts/train_conflict_detector.py` | Approach B MLP training |
| `scripts/validate_acc.py` | ACC Approach A validation |
| `configs/desktop_qlora.yaml` | Desktop RTX 3080 config |
| `configs/jetson_qlora.yaml` | Jetson Orin Nano config |
| `paper/draft.md` | Full paper draft (~3,000+ words) |

---

*Good morning! Coffee first, then GPU training. ☕🚀*
