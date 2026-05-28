"""Direct HTTP download of distilgpt2 with resume support."""

import os
from pathlib import Path
import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/distilgpt2/resolve/main"
LOCAL_DIR = Path("D:/ACC LLM Enhancement/models/distilgpt2")
CHUNK_SIZE = 8192

FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]

def download_file(filename, resume=True):
    url = f"{BASE_URL}/{filename}"
    target = LOCAL_DIR / filename
    
    existing_size = target.stat().st_size if target.exists() else 0
    
    headers = {}
    if resume and existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
        print(f"[{filename}] Resuming from {existing_size / (1024**2):.1f} MB")
    else:
        print(f"[{filename}] Starting download...")
    
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            if 'content-range' in r.headers:
                total_size = int(r.headers['content-range'].split('/')[-1])
            
            mode = 'ab' if resume and existing_size > 0 else 'wb'
            initial = existing_size if mode == 'ab' else 0
            
            with tqdm(total=total_size, initial=initial, unit='B', unit_scale=True, unit_divisor=1024, desc=filename[:20]) as pbar:
                with open(target, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        final_size = target.stat().st_size
        print(f"[{filename}] Done: {final_size / (1024**2):.1f} MB")
        return True
        
    except Exception as e:
        print(f"[{filename}] FAILED: {e}")
        return False

if __name__ == "__main__":
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        download_file(fname, resume=True)
        print()
