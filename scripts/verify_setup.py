"""Verify the ACC LLM environment is correctly configured for your hardware.

Run after installing requirements:
    .venv\Scripts\activate
    python scripts/verify_setup.py
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_device():
    """Detect and print available compute devices."""
    print("=" * 60)
    print("Device Check")
    print("=" * 60)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"✓ Intel XPU available: {torch.xpu.get_device_name(0)}")
        print(f"  Total memory: {torch.xpu.get_device_properties(0).total_memory // 1024**2} MB")
        return "xpu"
    elif torch.cuda.is_available():
        print(f"✓ NVIDIA CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        return "cuda"
    else:
        print("⚠ No GPU detected — using CPU (slow for 7B models)")
        return "cpu"


def check_packages():
    """Verify key packages are importable."""
    print("\n" + "=" * 60)
    print("Package Check")
    print("=" * 60)

    packages = [
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "wandb",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — MISSING")


def check_model_load(device: str):
    """Try loading tiny-gpt2 on the target device."""
    print("\n" + "=" * 60)
    print("Model Load Smoke Test (tiny-gpt2)")
    print("=" * 60)

    try:
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tok.pad_token = tok.eos_token

        if device == "xpu":
            model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
            model = model.to("xpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "sshleifer/tiny-gpt2",
                device_map=device if device != "cpu" else "cpu",
            )

        print(f"  ✓ Model loaded on {next(model.parameters()).device}")

        # Quick generation test
        inputs = tok("Hello", return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=3)
        text = tok.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ Generation works: '{text}'")

    except Exception as exc:
        print(f"  ✗ Model load/generation failed: {exc}")


def main():
    print("ACC LLM Environment Verification")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    device = check_device()
    check_packages()
    check_model_load(device)

    print("\n" + "=" * 60)
    if device in ("xpu", "cuda"):
        print("Environment looks good! Ready to load Mistral 7B.")
    else:
        print("Environment works, but CPU-only. Consider installing a GPU backend.")
    print("=" * 60)


if __name__ == "__main__":
    main()
