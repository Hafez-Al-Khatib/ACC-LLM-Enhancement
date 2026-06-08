"""Download and inspect HaluEval dataset for training hallucination detectors."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

DATA_DIR = Path("data/halueval")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_halueval():
    print("Downloading HaluEval dataset...")

    # Try common HaluEval dataset names on HF
    dataset_specs = [
        ("levertco/HaluEval", None),
        ("pminervini/HaluEval", "qa"),
        ("pminervini/HaluEval", "dialogue"),
        ("pminervini/HaluEval", "summarization"),
        ("pminervini/HaluEval", "general"),
        ("saucam/HaluEval", None),
    ]

    ds = None
    for name, config in dataset_specs:
        try:
            print(f"  Trying: {name}" + (f" (config={config})" if config else ""))
            kwargs = {"trust_remote_code": True} if config is None else {"trust_remote_code": True}
            ds = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
            print(f"  SUCCESS: loaded {name}" + (f" config={config}" if config else ""))
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if ds is None:
        raise RuntimeError("Could not load HaluEval from any known source")

    print("\nDataset structure:")
    print(ds)

    # Save to local JSONL files
    for split, dataset in ds.items():
        out_file = DATA_DIR / f"{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"  Saved {split}: {len(dataset)} examples -> {out_file}")

    # Inspect first few examples
    print("\n=== First example from each split ===")
    for split, dataset in ds.items():
        example = dataset[0]
        print(f"\n{split}:")
        for key, val in example.items():
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {key}: {val_str}")

    # Label distribution
    print("\n=== Label distributions ===")
    for split, dataset in ds.items():
        labels = []
        label_keys = [k for k in dataset[0].keys() if "label" in k.lower() or "halluc" in k.lower()]
        for key in label_keys:
            counts = Counter(str(ex.get(key)) for ex in dataset)
            print(f"\n{split}.{key}: {dict(counts)}")

    return ds


if __name__ == "__main__":
    download_halueval()
    print(f"\nAll data saved to: {DATA_DIR}")
