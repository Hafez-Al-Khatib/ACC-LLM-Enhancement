"""Standalone script for running ACC evaluation on Google Colab GPU.

Usage in Colab:
    !python colab_runner.py

This script:
1. Mounts Google Drive
2. Downloads Qwen2.5-1.5B if not present
3. Runs unified evaluation
4. Saves results to Drive
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
RESULTS_DIR = Path("/content/drive/MyDrive/ACC-LLM-Results")
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 15  # Can increase on GPU
DEVICE = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"


def mount_drive():
    """Mount Google Drive."""
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR}")


def setup_environment():
    """Install dependencies and clone repo."""
    os.chdir("/content")
    if not Path("ACC-LLM-Enhancement").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/YOUR_USERNAME/ACC-LLM-Enhancement.git"],
            check=True,
        )
    os.chdir("ACC-LLM-Enhancement")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "accelerate", "scipy", "scikit-learn"],
        check=True,
    )
    print("Environment ready.")


def download_model():
    """Download or copy model."""
    model_dir = Path("models/qwen2.5-1.5b")
    drive_model = Path("/content/drive/MyDrive/ACC-LLM-Models/qwen2.5-1.5b")

    if drive_model.exists():
        import shutil
        shutil.copytree(drive_model, model_dir, dirs_exist_ok=True)
        print("Model copied from Drive.")
    elif not model_dir.exists():
        from huggingface_hub import snapshot_download
        model_dir.mkdir(parents=True)
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Model downloaded to {model_dir}")
    else:
        print("Model already exists.")


def run_evaluation():
    """Run unified evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"eval_log_{timestamp}.txt"

    # Modify evaluation script to use GPU and more samples
    eval_script = Path("scripts/evaluate_all_methods.py")
    content = eval_script.read_text()
    content = content.replace('DEVICE = "cpu"', f'DEVICE = "{DEVICE}"')
    content = content.replace("MAX_NEW_TOKENS = 12", f"MAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
    eval_script.write_text(content)

    print(f"Running evaluation on {DEVICE}...")
    print(f"Log: {log_file}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [sys.executable, str(eval_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        process.wait()

    print(f"\nEvaluation finished (exit code: {process.returncode})")
    return process.returncode == 0


def save_results():
    """Copy results to Drive."""
    import shutil
    local_results = Path("results")
    if local_results.exists():
        for f in local_results.glob("*"):
            shutil.copy(f, RESULTS_DIR / f.name)
            print(f"Saved {f.name}")


def main():
    print("=" * 60)
    print("ACC Evaluation — Colab Runner")
    print("=" * 60)

    mount_drive()
    setup_environment()
    download_model()

    success = run_evaluation()
    save_results()

    if success:
        print("\nAll results saved to:", RESULTS_DIR)
    else:
        print("\nEvaluation failed. Check logs above.")


if __name__ == "__main__":
    main()
