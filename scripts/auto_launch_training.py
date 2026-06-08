"""
Auto-launch training when Mistral model shards are ready.

This script polls the model directory, verifies shard sizes,
runs a quick load test, and then launches QLoRA training.
Designed to be started while model is still downloading.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
import datetime

# ─── Config ──────────────────────────────────────────────────────────
MODEL_DIR = Path("models/mistral_7b")
SHARDS = {
    "model-00001-of-00003.safetensors": 4_949_453_792,
    "model-00002-of-00003.safetensors": 33_415_168,
    "model-00003-of-00003.safetensors": 9_513_178_144,
}
TRAIN_CONFIG = "configs/desktop_qlora.yaml"
RESULTS_DIR = Path("results")
ADAPTER_DIR = Path("adapters/desktop_run")

# Minimum file size to consider a shard "present" (not just a lock file)
MIN_SIZE_BYTES = 1_000_000

# Polling interval
POLL_INTERVAL_SECS = 60

# ─── Logging ─────────────────────────────────────────────────────────
def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"auto_launch_{datetime.date.today().isoformat()}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    # Force flush on every log
    class FlushHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    console = FlushHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, console],
    )
    return log_path


# ─── Shard Verification ──────────────────────────────────────────────
def check_shards():
    """Check which shards exist and are large enough."""
    ready = []
    missing = []
    incomplete = []
    for shard, expected in SHARDS.items():
        path = MODEL_DIR / shard
        if not path.exists():
            missing.append(shard)
            continue
        actual = path.stat().st_size
        if actual < MIN_SIZE_BYTES:
            incomplete.append(f"{shard}: {actual} bytes (too small)")
        else:
            ready.append(f"{shard}: {actual/1e9:.2f} GB")
    return ready, missing, incomplete


def verify_shards(tolerance=0.90):
    """Strict verification before launching training."""
    for shard, expected in SHARDS.items():
        path = MODEL_DIR / shard
        if not path.exists():
            return False, f"missing {shard}"
        actual = path.stat().st_size
        if actual < expected * tolerance:
            return False, f"{shard} incomplete: {actual/1e9:.2f}/{expected/1e9:.2f} GB"
    return True, "all shards verified"


# ─── Quick Load Test ─────────────────────────────────────────────────
def quick_load_test():
    """Try loading model+tokenizer to catch corruption early."""
    logging.info("Running quick model load test...")
    result = subprocess.run(
        [sys.executable, "scripts/validate_model_load.py"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if "MODEL_LOAD_OK" in result.stdout:
        logging.info("Model load test PASSED")
        return True
    logging.error("Model load test FAILED:\nstdout=%s\nstderr=%s", result.stdout, result.stderr)
    return False


# ─── Training Launcher ───────────────────────────────────────────────
def launch_training():
    """Launch training and stream logs."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_log = RESULTS_DIR / f"train_{timestamp}.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/train.py", "--config", TRAIN_CONFIG]
    logging.info("=" * 60)
    logging.info("LAUNCHING TRAINING")
    logging.info("Command: %s", " ".join(cmd))
    logging.info("Log file: %s", train_log)
    logging.info("=" * 60)

    with open(train_log, "w", encoding="utf-8") as fh:
        fh.write(f"=== ACC LLM Training {timestamp} ===\n")
        fh.write(f"Command: {' '.join(cmd)}\n")
        fh.write("=" * 60 + "\n")
        fh.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            # Log important lines at INFO level
            if any(k in lower for k in ("epoch", "loss", "nan", "error", "traceback", "saved", "training completed")):
                logging.info("[TRAIN] %s", line)

        proc.wait()
        fh.write(f"\n=== Exit code {proc.returncode} at {datetime.datetime.now().isoformat()} ===\n")

    return proc.returncode, train_log


# ─── Main ────────────────────────────────────────────────────────────
def main():
    master_log = setup_logging()
    logging.info("=" * 60)
    logging.info("ACC LLM Auto-Launcher Started")
    logging.info("Master log: %s", master_log)
    logging.info("Polling every %d seconds for model shards...", POLL_INTERVAL_SECS)

    # Wait for shards to appear
    while True:
        ready, missing, incomplete = check_shards()
        if ready and not missing:
            logging.info("All shards present! Ready=%s", ready)
            break
        msg_parts = []
        if missing:
            msg_parts.append(f"missing={missing}")
        if incomplete:
            msg_parts.append(f"incomplete={incomplete}")
        if ready:
            msg_parts.append(f"ready={ready}")
        logging.info("Waiting for model... %s", "; ".join(msg_parts) if msg_parts else "checking...")
        time.sleep(POLL_INTERVAL_SECS)

    # Strict verification
    ok, msg = verify_shards()
    if not ok:
        logging.error("Shard verification failed: %s", msg)
        return 1
    logging.info("Shard verification passed: %s", msg)

    # Quick load test
    if not quick_load_test():
        logging.error("Model load test failed. Aborting.")
        return 1

    # Launch training
    returncode, train_log = launch_training()

    if returncode == 0:
        logging.info("=" * 60)
        logging.info("Training completed successfully!")
        logging.info("Adapter: %s", ADAPTER_DIR)
        logging.info("Train log: %s", train_log)
        logging.info("=" * 60)
    else:
        logging.error("=" * 60)
        logging.error("Training failed with exit code %d", returncode)
        logging.error("Train log: %s", train_log)
        logging.error("=" * 60)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
