"""Pre-download distilgpt2 to local cache."""
from huggingface_hub import snapshot_download
import os

REPO_ID = "distilgpt2"
LOCAL_DIR = "models/distilgpt2"

print(f"Downloading {REPO_ID} to {LOCAL_DIR}...")
os.makedirs(LOCAL_DIR, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
)

print("Done!")
for f in sorted(os.listdir(LOCAL_DIR)):
    size_mb = os.path.getsize(os.path.join(LOCAL_DIR, f)) / (1024**2)
    print(f"  {f}: {size_mb:.1f} MB")
