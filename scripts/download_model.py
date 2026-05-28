"""Download model weights using huggingface_hub directly with resume support."""

from huggingface_hub import snapshot_download
import os

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
local_dir = "models/mistral_7b"

try:
    print(f"Downloading {repo_id} to {local_dir}...")
    print("This will download ~15GB of model weights.")
    print("Progress will be slow — please be patient.")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Download complete!")
    
    # List what we got
    for f in sorted(os.listdir(local_dir)):
        size = os.path.getsize(os.path.join(local_dir, f)) / (1024**3)
        print(f"  {f}: {size:.2f} GB")
        
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
