"""Direct HTTP download of missing Mistral shards with resume support."""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LOCAL_DIR = Path("models/mistral_7b")
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

CHUNK_SIZE = 8192  # 8KB chunks

def get_file_size(filename):
    """Get expected file size from HEAD request."""
    url = f"{BASE_URL}/{filename}"
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200:
            return int(r.headers.get('content-length', 0))
    except Exception as e:
        print(f"HEAD request failed: {e}")
    return None

def download_file(filename, resume=True):
    """Download a single file with resume support and progress bar."""
    url = f"{BASE_URL}/{filename}"
    target = LOCAL_DIR / filename
    
    # Get expected size
    expected_size = get_file_size(filename)
    if expected_size:
        print(f"Expected size: {expected_size / (1024**3):.2f} GB")
    
    # Check existing file
    existing_size = target.stat().st_size if target.exists() else 0
    if existing_size > 0:
        print(f"Existing file: {existing_size / (1024**3):.2f} GB")
        if expected_size and existing_size >= expected_size:
            print(f"[OK] {filename} already complete!")
            return True
        if not resume:
            print("Removing incomplete file...")
            target.unlink()
            existing_size = 0
    
    # Set up resume headers
    headers = {}
    if resume and existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
        print(f"Resuming from {existing_size / (1024**3):.2f} GB")
    
    # Start download
    print(f"Downloading from {url}...")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            
            # Determine total size
            total_size = expected_size
            if 'content-range' in r.headers:
                # Parse total from range header
                total_size = int(r.headers['content-range'].split('/')[-1])
            elif 'content-length' in r.headers:
                total_size = int(r.headers['content-length']) + existing_size
            
            # Progress bar
            mode = 'ab' if resume and existing_size > 0 else 'wb'
            initial = existing_size if mode == 'ab' else 0
            
            with tqdm(total=total_size, initial=initial, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with open(target, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        # Verify
        final_size = target.stat().st_size
        print(f"Downloaded: {final_size / (1024**3):.2f} GB")
        if expected_size and final_size != expected_size:
            print(f"[WARN] Size mismatch! Expected {expected_size}, got {final_size}")
            return False
        print(f"[OK] {filename} complete!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Download error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Direct HTTP Download - Mistral 7B Shards")
    print("=" * 60)
    
    all_ok = True
    for shard in SHARDS:
        print(f"\n{'='*60}")
        print(f"Shard: {shard}")
        print('='*60)
        
        ok = download_file(shard, resume=True)
        if not ok:
            all_ok = False
            print(f"FAILED: {shard}")
    
    print(f"\n{'='*60}")
    if all_ok:
        print("[SUCCESS] All shards downloaded!")
    else:
        print("[FAILURE] Some shards failed.")
    print('='*60)

if __name__ == "__main__":
    main()
