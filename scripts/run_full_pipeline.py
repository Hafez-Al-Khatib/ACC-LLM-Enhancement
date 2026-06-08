#!/usr/bin/env python3
"""
Full ACC LLM Pipeline — Auto-runs all steps after model download completes.

Usage:
    # Start monitoring and auto-run when model is ready
    python scripts/run_full_pipeline.py --config configs/desktop_qlora.yaml

    # Or run immediately (if model already downloaded)
    python scripts/run_full_pipeline.py --config configs/desktop_qlora.yaml --no-wait

Steps:
    1. Wait for model download
    2. Verify model loads
    3. Generate calibration prompts
    4. QLoRA fine-tuning
    5. Generate conflict detector training data
    6. Train conflict detector
    7. Run ACC validation
    8. Aggregate results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL_DIR = "models/mistral_7b"
EXPECTED_SHARDS = [
    ("model-00001-of-00003.safetensors", 4_900_000_000),
    ("model-00002-of-00003.safetensors", 30_000_000),
    ("model-00003-of-00003.safetensors", 8_900_000_000),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_cmd(cmd, cwd=None, timeout=None):
    """Run a command and return success status."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            log(f"ERROR (exit {result.returncode}): {result.stderr[:500]}")
            return False
        log("SUCCESS")
        return True
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        log(f"EXCEPTION: {e}")
        return False


def wait_for_model():
    """Poll until all model shards are present."""
    log("Waiting for Mistral 7B model download...")
    while True:
        all_complete = True
        for filename, expected_size in EXPECTED_SHARDS:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                size = os.path.getsize(path)
                pct = 100 * size / expected_size
                status = f"{size/1e9:.2f} GB ({pct:.1f}%)"
                if pct < 99:
                    all_complete = False
            else:
                status = "NOT STARTED"
                all_complete = False
            log(f"  {filename}: {status}")
        
        if all_complete:
            log("All model shards downloaded!")
            return True
        
        log("Waiting 60s...")
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Full ACC LLM pipeline")
    parser.add_argument("--config", default="configs/desktop_qlora.yaml", help="Training config")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for download")
    parser.add_argument("--skip-training", action="store_true", help="Skip QLoRA training (use existing adapter)")
    args = parser.parse_args()

    pipeline_log = "results/pipeline_log.jsonl"
    os.makedirs("results", exist_ok=True)

    def record_step(step, success, notes=""):
        with open(pipeline_log, "a") as f:
            f.write(json.dumps({
                "step": step,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "notes": notes,
            }) + "\n")

    # Step 1: Wait for model
    if not args.no_wait:
        success = wait_for_model()
        record_step("wait_for_model", success)
        if not success:
            sys.exit(1)

    # Step 2: Verify model
    log("Step 2: Verifying model...")
    success = run_cmd([sys.executable, "scripts/validate_model_load.py"], timeout=300)
    record_step("validate_model", success)
    if not success:
        sys.exit(1)

    # Step 3: Generate calibration prompts
    log("Step 3: Generating calibration prompts...")
    datasets = [
        ("experiments/datasets/pubmedqa/train.jsonl", "data/calibration/pubmedqa_cal.jsonl"),
        ("experiments/datasets/sciq/train.jsonl", "data/calibration/sciq_cal.jsonl"),
    ]
    for ds_path, out_path in datasets:
        if os.path.exists(ds_path):
            run_cmd([
                sys.executable, "scripts/generate_calibration_prompts.py",
                "--dataset", ds_path,
                "--output", out_path,
                "--num-prompts", "100",
            ], timeout=60)
    record_step("generate_calibration", True)

    # Step 4: QLoRA training
    adapter_path = None
    if not args.skip_training:
        log("Step 4: QLoRA training...")
        success = run_cmd([sys.executable, "scripts/train.py", "--config", args.config], timeout=3600*6)
        record_step("qlora_training", success)
        if success:
            # Extract output dir from config
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            adapter_path = os.path.join(cfg["training"]["output_dir"], "final_adapter")
    else:
        log("Step 4: Skipping training (using existing adapter)")
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        adapter_path = os.path.join(cfg["training"]["output_dir"], "final_adapter")

    # Step 5: Generate conflict data
    if adapter_path and os.path.exists(adapter_path):
        log("Step 5: Generating conflict detector training data...")
        success = run_cmd([
            sys.executable, "scripts/generate_conflict_data.py",
            "--model", MODEL_DIR,
            "--adapter", adapter_path,
            "--output", "data/acc_training/mistral_conflict_data.jsonl",
            "--num_samples_per_class", "500",
            "--max_new_tokens", "20",
        ], timeout=3600)
        record_step("generate_conflict_data", success)

        # Step 6: Train conflict detector
        log("Step 6: Training conflict detector...")
        success = run_cmd([
            sys.executable, "scripts/train_conflict_detector.py",
            "--data", "data/acc_training/mistral_conflict_data.jsonl",
            "--save_dir", "adapters/acc_conflict_detector",
            "--hidden_ratio", "0.5",
            "--dropout", "0.1",
            "--epochs", "100",
            "--patience", "10",
        ], timeout=1800)
        record_step("train_conflict_detector", success)

    # Step 7: ACC validation
    log("Step 7: Running ACC validation...")
    test_datasets = [
        ("experiments/datasets/pubmedqa/test.jsonl", "results/pubmedqa_eval.json"),
        ("experiments/datasets/sciq/test.jsonl", "results/sciq_eval.json"),
    ]
    for ds_path, out_path in test_datasets:
        if os.path.exists(ds_path):
            run_cmd([
                sys.executable, "scripts/validate_acc.py",
                "--adapter", adapter_path or "adapters/desktop_run/final_adapter",
                "--config", args.config,
                "--dataset", ds_path,
                "--output", out_path,
            ], timeout=3600)
    record_step("acc_validation", True)

    # Step 8: Aggregate results
    log("Step 8: Aggregating results...")
    run_cmd([
        sys.executable, "scripts/aggregate_results.py",
        "--input", "results/",
        "--output", "results/summary/",
    ], timeout=300)
    record_step("aggregate_results", True)

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log(f"Log: {pipeline_log}")
    log("=" * 60)


if __name__ == "__main__":
    main()
