"""Setup script for RTX 4090 experiments.

Run once on the 4090 machine to install deps and download models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], check: bool = True):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Model to download")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if not args.skip_deps:
        print("\n=== Installing dependencies ===")
        run([sys.executable, "-m", "pip", "install", "-r", str(repo_root / "requirements-cuda.txt")])

    print("\n=== Verifying CUDA ===")
    run([sys.executable, "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"])

    print(f"\n=== Downloading model: {args.model} ===")
    from huggingface_hub import snapshot_download

    model_name_safe = args.model.replace("/", "_")
    model_dir = repo_root / "models" / model_name_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to: {model_dir}")

    print("\n=== Setup complete ===")
    print(f"To run evaluation: python scripts/run_4090_eval.py --model models/{model_name_safe}")


if __name__ == "__main__":
    main()
