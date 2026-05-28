"""Alternative download approach using snapshot_download with explicit progress."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LOCAL_DIR = Path("D:/ACC LLM Enhancement/models/mistral_7b")

def download_with_progress():
    """Download model with explicit progress reporting."""
    print(f"Downloading {REPO_ID}...")
    print(f"Target directory: {LOCAL_DIR}")
    print("This will download ~15GB total (3 shards)")
    print("Progress updates every 30 seconds...\n")
    
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("\nDownload complete!")
        
        # Verify files
        for f in sorted(os.listdir(LOCAL_DIR)):
            if f.endswith('.safetensors'):
                size_gb = os.path.getsize(os.path.join(LOCAL_DIR, f)) / (1024**3)
                print(f"  {f}: {size_gb:.2f} GB")
                
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_with_progress()
