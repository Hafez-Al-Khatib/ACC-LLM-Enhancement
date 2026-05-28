"""Quick validation: can we load the quantized model + tokenizer?"""
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import load_model, load_tokenizer, make_bnb_config
import yaml
import torch

def main():
    config_path = ROOT / "configs" / "desktop_qlora.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bnb = make_bnb_config(cfg["quantization"])
    dtype_name = cfg["model"].get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name)

    model = load_model(
        cfg["model"]["base_model"],
        bnb_config=bnb,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    tok = load_tokenizer(cfg["model"]["base_model"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL_LOAD_OK params={total_params:,} trainable={trainable:,}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
