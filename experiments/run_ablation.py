"""Batch experiment runner for systematic ablation studies.

Example usage:
    python experiments/run_ablation.py \
        --vertical medical \
        --dataset pubmedqa \
        --ablate rank \
        --values 4 8 16 32 \
        --hardware desktop \
        --config-template configs/desktop_qlora.yaml

This generates one config per value, runs training sequentially, and logs
results to experiments/results/{vertical}_{dataset}_{timestamp}.jsonl.
"""

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def generate_config(base_config: dict, ablate_key: str, value, exp_name: str) -> dict:
    """Create a derived config with one hyperparameter changed."""
    cfg = copy.deepcopy(base_config)

    # Map ablate_key to config path
    key_map = {
        "rank": ("lora", "r"),
        "alpha": ("lora", "lora_alpha"),
        "lr": ("training", "learning_rate"),
        "seq_length": ("training", "max_seq_length"),
        "batch_size": ("training", "per_device_train_batch_size"),
        "accum": ("training", "gradient_accumulation_steps"),
        "dropout": ("lora", "lora_dropout"),
        "acc_threshold": ("acc", "threshold"),
        "acc_mode": ("acc", "mode"),
        "self_consistency_samples": ("acc", "self_consistency_samples"),
    }

    if ablate_key not in key_map:
        raise ValueError(f"Unknown ablation key: {ablate_key!r}. Available: {list(key_map)}")

    section, param = key_map[ablate_key]

    def _infer_type(v):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    cfg[section][param] = _infer_type(value)

    # Update output dir and run name
    cfg["training"]["output_dir"] = f"adapters/{exp_name}"
    if "wandb" in cfg:
        cfg["wandb"]["run_name"] = exp_name

    return cfg


def run_single_experiment(config_path: str, results_file: str):
    """Execute one training run and append metadata to results."""
    start = time.time()
    logger.info("▶ Starting: %s", config_path)

    try:
        proc = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", config_path],
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour max per run
        )
        elapsed = time.time() - start
        success = proc.returncode == 0

        result = {
            "config": config_path,
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }

        if not success:
            logger.error("✗ FAILED: %s (exit code %d)", config_path, proc.returncode)
        else:
            logger.info("✓ DONE: %s in %.1fs", config_path, elapsed)

    except subprocess.TimeoutExpired:
        result = {
            "config": config_path,
            "success": False,
            "elapsed_seconds": 3600 * 4,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Timeout: exceeded 4 hours",
        }
        logger.error("✗ TIMEOUT: %s", config_path)

    # Append to results file
    with open(results_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")

    return result["success"]


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--vertical", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ablate", required=True, help="Hyperparameter to ablate")
    parser.add_argument("--values", nargs="+", required=True, help="Values to try")
    parser.add_argument("--config-template", required=True, help="Base YAML config")
    parser.add_argument("--hardware", default="desktop", choices=["desktop", "jetson"])
    parser.add_argument("--dataset-dir", default=None, help="Pre-built dataset dir")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--eval-after-train", action="store_true", help="Run compare_baselines.py after each successful training run")
    args = parser.parse_args()

    # Load base config
    with open(args.config_template, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    # Auto-download dataset if not provided
    dataset_dir = args.dataset_dir or f"experiments/datasets/{args.dataset}"
    if not Path(dataset_dir).exists():
        logger.info("Dataset not found locally — downloading...")
        subprocess.run(
            [sys.executable, "experiments/datasets/auto_load.py",
             "--dataset", args.dataset,
             "--output-dir", dataset_dir]
            + ([f"--max-samples={args.max_samples}"] if args.max_samples else []),
            check=True,
        )

    # Update base config with dataset paths
    base_cfg["dataset"] = {
        "path": str(Path(dataset_dir) / "train.jsonl"),
        "eval_path": str(Path(dataset_dir) / "val.jsonl"),
        "test_path": str(Path(dataset_dir) / "test.jsonl"),
        "text_column": "text",
        "format": "jsonl",
    }

    if args.epochs:
        base_cfg["training"]["num_train_epochs"] = args.epochs

    # Results file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{args.vertical}_{args.dataset}_{args.ablate}_{ts}.jsonl"

    logger.info("=" * 60)
    logger.info("ABLATION: %s on %s | %s runs", args.ablate, args.dataset, len(args.values))
    logger.info("Results → %s", results_file)
    logger.info("=" * 60)

    for i, val in enumerate(args.values):
        exp_name = f"{args.vertical}_{args.dataset}_{args.ablate}{val}_{args.hardware}_{ts}"
        cfg = generate_config(base_cfg, args.ablate, val, exp_name)

        cfg_path = results_dir / f"{exp_name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        logger.info("\n[%d/%d] %s = %s", i + 1, len(args.values), args.ablate, val)
        success = run_single_experiment(str(cfg_path), str(results_file))

        if success and args.eval_after_train:
            logger.info("Running post-training evaluation...")
            test_path = str(Path(dataset_dir) / "test.jsonl")
            eval_dir = results_dir / f"{exp_name}_eval"
            try:
                subprocess.run(
                    [sys.executable, "experiments/compare_baselines.py",
                     "--config", str(cfg_path),
                     "--prompts", test_path,
                     "--output_dir", str(eval_dir),
                     "--max_new_tokens", str(base_cfg.get("inference", {}).get("max_new_tokens", 128)),
                     "--temperature", str(base_cfg.get("inference", {}).get("temperature", 0.7))],
                    check=True,
                    timeout=3600,
                )
                logger.info("Evaluation complete → %s", eval_dir)
            except Exception as exc:
                logger.warning("Evaluation failed for %s: %s", exp_name, exc)

        if not success:
            logger.warning("Run failed — continuing to next value...")

    logger.info("\nAll runs complete. Results: %s", results_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
