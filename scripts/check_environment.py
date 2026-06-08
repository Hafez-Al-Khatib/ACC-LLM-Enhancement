"""Environment readiness checker for ACC LLM training.

Run this before starting experiments to catch setup issues early.
"""

import sys
from pathlib import Path

_CHECKS = []


def check(name):
    def decorator(fn):
        _CHECKS.append((name, fn))
        return fn
    return decorator


@check("Python version")
def check_python():
    ok = sys.version_info >= (3, 10)
    return ok, f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@check("PyTorch + CUDA")
def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        version = torch.__version__
        cuda_ver = torch.version.cuda or "N/A"
        return cuda, f"PyTorch {version} | CUDA available: {cuda} | CUDA version: {cuda_ver}"
    except Exception as e:
        return False, f"PyTorch import failed: {e}"


@check("Transformers")
def check_transformers():
    try:
        import transformers
        return True, f"transformers {transformers.__version__}"
    except Exception as e:
        return False, str(e)


@check("PEFT")
def check_peft():
    try:
        import peft
        return True, f"peft {peft.__version__}"
    except Exception as e:
        return False, str(e)


@check("BitsAndBytes (4-bit support)")
def check_bitsandbytes():
    try:
        import bitsandbytes
        return True, f"bitsandbytes {bitsandbytes.__version__}"
    except Exception as e:
        return False, str(e)


@check("Datasets")
def check_datasets():
    try:
        import datasets
        return True, f"datasets {datasets.__version__}"
    except Exception as e:
        return False, str(e)


@check("SciPy (for statistical tests)")
def check_scipy():
    try:
        import scipy
        return True, f"scipy {scipy.__version__}"
    except Exception as e:
        return False, str(e)


@check("Scikit-learn (for conflict detector metrics)")
def check_sklearn():
    try:
        import sklearn
        return True, f"scikit-learn {sklearn.__version__}"
    except Exception as e:
        return False, str(e)


@check("Mistral 7B model files")
def check_model():
    model_dir = Path("models/mistral_7b")
    required = [
        "config.json",
        "tokenizer.model",
        "model.safetensors.index.json",
    ]
    shards = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    missing = [f for f in required if not (model_dir / f).exists()]
    missing_shards = [f for f in shards if not (model_dir / f).exists()]
    if missing:
        return False, f"Missing required files: {missing}"
    if missing_shards:
        return False, f"Missing model shards: {missing_shards} (download in progress?)"
    return True, f"All model files present in {model_dir}"


@check("Training datasets")
def check_datasets_local():
    datasets = {
        "PubMedQA": Path("experiments/datasets/pubmedqa/train.jsonl"),
        "SciQ": Path("experiments/datasets/sciq/train.jsonl"),
        "General Instruction": Path("experiments/datasets/general_instruction/train.jsonl"),
    }
    missing = [name for name, path in datasets.items() if not path.exists()]
    present = [name for name, path in datasets.items() if path.exists()]
    if missing:
        return False, f"Missing: {missing}. Present: {present}"
    return True, f"All datasets ready: {present}"


@check("GPU memory (if CUDA available)")
def check_gpu_memory():
    try:
        import torch
        if not torch.cuda.is_available():
            return True, "No CUDA — skipping GPU memory check"
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total - reserved
        ok = total >= 8  # Need at least 8GB for QLoRA
        return ok, f"GPU 0: {total:.1f} GB total, {free:.1f} GB free"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("ACC LLM Environment Check")
    print("=" * 60)

    all_ok = True
    for name, fn in _CHECKS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"EXCEPTION: {e}"
        status = "[PASS]" if ok else "[FAIL]"
        print(f"\n{status} — {name}")
        print(f"   {msg}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed! Ready for training.")
    else:
        print("Some checks failed. Fix the issues above before training.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
