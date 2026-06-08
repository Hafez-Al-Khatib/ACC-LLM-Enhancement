#!/usr/bin/env python3
"""
Monitor Mistral 7B download progress and report status.

Usage:
    python scripts/monitor_download.py
"""

import os
import time
import glob
from pathlib import Path

MODEL_DIR = "models/mistral_7b"
EXPECTED_FILES = {
    "model-00001-of-00003.safetensors": 4_900_000_000,
    "model-00002-of-00003.safetensors": 30_000_000,
    "model-00003-of-00003.safetensors": 8_900_000_000,
}


def get_file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main():
    print("Monitoring Mistral 7B download progress...")
    print(f"Target directory: {MODEL_DIR}")
    print(f"Expected total: ~{sum(EXPECTED_FILES.values()) / 1e9:.1f} GB")
    print()
    
    while True:
        total_downloaded = 0
        total_expected = 0
        all_complete = True
        
        print(f"[{time.strftime('%H:%M:%S')}] Status:")
        for filename, expected_size in EXPECTED_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            size = get_file_size(path)
            total_downloaded += size
            total_expected += expected_size
            
            if size >= expected_size * 0.99:
                status = "COMPLETE"
            elif size > 0:
                status = f"DOWNLOADING {size/1e9:.2f} GB / {expected_size/1e9:.2f} GB ({100*size/expected_size:.1f}%)"
                all_complete = False
            else:
                status = "NOT STARTED"
                all_complete = False
            
            print(f"  {filename}: {status}")
        
        overall_pct = 100 * total_downloaded / total_expected if total_expected > 0 else 0
        print(f"  Overall: {total_downloaded/1e9:.2f} GB / {total_expected/1e9:.2f} GB ({overall_pct:.1f}%)")
        
        if all_complete:
            print("\nALL FILES DOWNLOADED! Ready for training.")
            break
        
        print()
        time.sleep(30)


if __name__ == "__main__":
    main()
