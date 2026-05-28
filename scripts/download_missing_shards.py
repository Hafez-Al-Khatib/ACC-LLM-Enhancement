"""Robust download of missing Mistral shards with retry logic."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import time

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LOCAL_DIR = Path("D:/ACC LLM Enhancement/models/mistral_7b")
CACHE_DIR = LOCAL_DIR / ".cache/huggingface/download"

# Expected shards
SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

MAX_RETRIES = 3
TIMEOUT = 300  # 5 minutes per file

def check_existing():
    """Check which shards exist."""
    existing = {}
    for shard in SHARDS:
        path = LOCAL_DIR / shard
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            existing[shard] = size_gb
            print(f"  [OK] {shard}: {size_gb:.2f} GB")
        else:
            print(f"  [MISSING] {shard}: MISSING")
    return existing

def download_shard(filename: str, retries=MAX_RETRIES):
    """Download a single shard with retry."""
    target = LOCAL_DIR / filename
    
    for attempt in range(retries):
        try:
            print(f"\nDownloading {filename} (attempt {attempt+1}/{retries})...")
            
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(LOCAL_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            size_gb = Path(downloaded).stat().st_size / (1024**3)
            print(f"  [OK] Success: {size_gb:.2f} GB")
            return True
            
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            if attempt < retries - 1:
                wait = 2 ** attempt  # exponential backoff
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    
    return False

if __name__ == "__main__":
    print("Checking existing shards...")
    existing = check_existing()
    
    missing = [s for s in SHARDS if s not in existing]
    
    if not missing:
        print("\nAll shards present! Ready for training.")
        exit(0)
    
    print(f"\nMissing {len(missing)} shard(s): {missing}")
    
    # Download each missing shard
    all_ok = True
    for shard in missing:
        ok = download_shard(shard)
        if not ok:
            all_ok = False
            print(f"  FAILED to download {shard}")
    
    if all_ok:
        print("\n[OK] All shards downloaded successfully!")
    else:
        print("\n[FAIL] Some shards failed to download. Alternative approach needed.")
        exit(1)
