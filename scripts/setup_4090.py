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
        print("\n=== Installing PyTorch with CUDA ===")
        run([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ])

        print("\n=== Installing remaining dependencies ===")
        run([sys.executable, "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])

    print("\n=== Verifying CUDA ===")
    run([sys.executable, "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"])

    print(f"\n=== Downloading model: {args.model} ===")
    print("Using robust downloader with retries and resume support...")

    model_name_safe = args.model.replace("/", "_")
    model_dir = repo_root / "models" / model_name_safe

    # Use robust downloader with retries
    run([
        sys.executable,
        str(repo_root / "scripts" / "download_model_robust_hf.py"),
        args.model,
        "--local-dir", str(model_dir),
        "--max-retries", "20",
        "--delay", "10",
    ])
    print(f"Model saved to: {model_dir}")

    print("\n=== Setup complete ===")
    print(f"To run evaluation: python scripts/run_4090_eval.py --model models/{model_name_safe}")


if __name__ == "__main__":
    main()
