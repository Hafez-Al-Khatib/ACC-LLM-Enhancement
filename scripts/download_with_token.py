import os
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'

from huggingface_hub import login
login(token='YOUR_HF_TOKEN_HERE')
print("HF login successful")

from huggingface_hub import hf_hub_download

files = [
    'model.safetensors',
    'vocab.json', 
    'tokenizer_config.json',
    'merges.txt',
    'config.json'
]

for fname in files:
    print(f"Downloading {fname}...")
    try:
        path = hf_hub_download(
            repo_id='distilgpt2',
            filename=fname,
            local_dir='models/distilgpt2',
            local_dir_use_symlinks=False,
        )
        size = os.path.getsize(path) / (1024**2)
        print(f"  OK: {size:.1f} MB")
    except Exception as e:
        print(f"  FAIL: {e}")

print("\nAll downloads done!")
