"""
ACC LLM Overnight Launcher + Monitor
=====================================
Waits for model download, verifies shards, launches training, and monitors GPU/logs.
"""
import os
import sys
import time
import json
import subprocess
import threading
import logging
from pathlib import Path
import datetime

# ─── Config ──────────────────────────────────────────────────────────
MODEL_DIR = Path("models/mistral_7b")
SHARDS = {
    "model-00001-of-00003.safetensors": 4_949_453_792,
    "model-00002-of-00003.safetensors": 33_415_168,
    "model-00003-of-00003.safetensors": 9_513_178_144,  # ~8.86 GB expected
}
TRAIN_CONFIG = "configs/desktop_qlora.yaml"
RESULTS_DIR = Path("results")
ADAPTER_DIR = Path("adapters/desktop_run")

# Monitor interval
GPU_POLL_SECS = 300  # 5 minutes
LOG_POLL_SECS = 60   # 1 minute

# ─── Logging ─────────────────────────────────────────────────────────
def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"overnight_master_{datetime.date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


# ─── Model Verification ─────────────────────────────────────────────
def verify_shards(tolerance: float = 0.95) -> tuple[bool, str]:
    """Check all shards exist and are reasonably sized."""
    missing = []
    incomplete = []
    for shard, expected in SHARDS.items():
        path = MODEL_DIR / shard
        if not path.exists():
            missing.append(shard)
            continue
        actual = path.stat().st_size
        if actual < expected * tolerance:
            incomplete.append(f"{shard}: {actual/1e9:.2f}/{expected/1e9:.2f} GB")
    if missing or incomplete:
        msg = f"missing={missing}, incomplete={incomplete}"
        return False, msg
    return True, "all shards OK"


def wait_for_model(timeout_hours: float = 2.0) -> bool:
    """Poll until model is ready or timeout."""
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        ok, msg = verify_shards()
        if ok:
            logging.info("Model shards verified: %s", msg)
            return True
        logging.info("Waiting for model… %s", msg)
        time.sleep(60)
    logging.error("Model download timed out after %.1f hours", timeout_hours)
    return False


# ─── Quick Load Test ─────────────────────────────────────────────────
def quick_load_test() -> bool:
    """Try to load the model+tokenizer to catch corruption early."""
    logging.info("Running quick model load test…")
    result = subprocess.run(
        [sys.executable, "scripts/validate_model_load.py"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    if "MODEL_LOAD_OK" in result.stdout:
        logging.info("Model load test passed: %s", result.stdout.strip())
        return True
    logging.error("Model load test FAILED:\nstdout=%s\nstderr=%s", result.stdout, result.stderr)
    return False


# ─── Training Launcher ───────────────────────────────────────────────
def launch_training() -> tuple[int, Path]:
    """Launch training and return (exit_code, log_file)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_log = RESULTS_DIR / f"overnight_train_{timestamp}.log"
    train_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/train.py", "--config", TRAIN_CONFIG]
    logging.info("Launching training → %s", train_log.name)
    logging.info("Command: %s", " ".join(cmd))

    with open(train_log, "w", encoding="utf-8") as fh:
        fh.write(f"=== ACC LLM Overnight Training {timestamp} ===\n")
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

        # Stream to log and sniff for key events
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            lower = line.lower()
            if any(k in lower for k in ("epoch", "loss", "nan", "error", "traceback", "saved")):
                logging.info("[TRAIN] %s", line.strip())

        proc.wait()
        fh.write(f"\n=== Exit code {proc.returncode} at {datetime.datetime.now().isoformat()} ===\n")

    return proc.returncode, train_log


# ─── GPU Monitor ─────────────────────────────────────────────────────
def gpu_monitor(stop_event: threading.Event):
    """Background thread: log GPU stats every 5 min."""
    gpu_log = RESULTS_DIR / f"overnight_gpu_{datetime.date.today().isoformat()}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gpu_log, "a", encoding="utf-8") as fh:
        fh.write("timestamp,memory_used_mib,memory_free_mib,util_pct,temp_c\n")

    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            now = datetime.datetime.now().isoformat()
            line = out.stdout.strip().replace(" ", "")
            with open(gpu_log, "a", encoding="utf-8") as fh:
                fh.write(f"{now},{line}\n")
        except Exception as e:
            logging.warning("GPU monitor error: %s", e)

        # Sleep in small increments so we respect stop_event quickly
        for _ in range(GPU_POLL_SECS // 5):
            if stop_event.is_set():
                break
            time.sleep(5)

    logging.info("GPU monitor stopped.")


# ─── Log Sniffer ────────────────────────────────────────────────────
def log_sniffer(train_log: Path, stop_event: threading.Event):
    """Background thread: watch training log for NaN / errors."""
    alert_log = RESULTS_DIR / f"overnight_alerts_{datetime.date.today().isoformat()}.log"
    last_size = 0

    while not stop_event.is_set():
        if not train_log.exists():
            time.sleep(LOG_POLL_SECS)
            continue

        current_size = train_log.stat().st_size
        if current_size <= last_size:
            time.sleep(LOG_POLL_SECS)
            continue

        with open(train_log, "r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(last_size)
            new_text = fh.read().lower()
            last_size = current_size

        alerts = []
        if "nan" in new_text:
            alerts.append("NaN detected")
        if "traceback" in new_text and "error" in new_text:
            alerts.append("Python traceback/error detected")
        if "cuda out of memory" in new_text:
            alerts.append("CUDA OOM")

        if alerts:
            now = datetime.datetime.now().isoformat()
            with open(alert_log, "a", encoding="utf-8") as fh:
                for a in alerts:
                    msg = f"{now} ALERT: {a}"
                    fh.write(msg + "\n")
                    logging.warning(msg)

        time.sleep(LOG_POLL_SECS)

    logging.info("Log sniffer stopped.")


# ─── Main ────────────────────────────────────────────────────────────
def main():
    master_log = setup_logging()
    logging.info("=" * 60)
    logging.info("ACC LLM Overnight Automation Started")
    logging.info("Master log: %s", master_log)

    # 1. Wait for model
    if not wait_for_model(timeout_hours=3.0):
        logging.error("Aborting — model not ready.")
        return 1

    # 2. Quick load test
    if not quick_load_test():
        logging.error("Aborting — model load test failed.")
        return 1

    # 3. Launch background monitor
    stop_event = threading.Event()
    gpu_thread = threading.Thread(target=gpu_monitor, args=(stop_event,), daemon=True)
    gpu_thread.start()

    # 4. Launch training
    returncode, train_log = launch_training()

    # 5. Stop monitor
    stop_event.set()
    gpu_thread.join(timeout=30)

    # 6. Final status
    if returncode == 0:
        logging.info("Training completed successfully.")
        logging.info("Adapter: %s", ADAPTER_DIR)
        logging.info("Train log: %s", train_log)
    else:
        logging.error("Training failed with exit code %d", returncode)
        logging.error("Train log: %s", train_log)

    logging.info("=" * 60)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
