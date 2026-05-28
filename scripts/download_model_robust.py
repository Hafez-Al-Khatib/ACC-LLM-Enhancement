"""Robust model download with per-file retry logic."""

from huggingface_hub import hf_hub_download
import os
import time

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
local_dir = "models/mistral_7b"
os.makedirs(local_dir, exist_ok=True)

files = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors", 
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
]

# Check what we already have
existing = {f: os.path.getsize(os.path.join(local_dir, f)) / (1024**3) 
            for f in files if os.path.exists(os.path.join(local_dir, f))}
print("Already downloaded:")
for f, s in existing.items():
    print(f"  {f}: {s:.2f} GB")

# Download missing files
for filename in files:
    filepath = os.path.join(local_dir, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100*1024*1024:  # > 100MB
        print(f"OK {filename} already exists ({os.path.getsize(filepath)/(1024**3):.2f} GB)")
        continue
    
    print(f"\nDownloading {filename}...")
    for attempt in range(3):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            size_gb = os.path.getsize(downloaded) / (1024**3)
            print(f"OK {filename}: {size_gb:.2f} GB")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
            time.sleep(5)
    else:
        print(f"FAIL Failed to download {filename} after 3 attempts")

print("\nDone. Verifying...")
total = 0
for f in files:
    p = os.path.join(local_dir, f)
    if os.path.exists(p):
        s = os.path.getsize(p)
        total += s
        print(f"  {f}: {s/(1024**3):.2f} GB")
print(f"\nTotal: {total/(1024**3):.2f} GB")
