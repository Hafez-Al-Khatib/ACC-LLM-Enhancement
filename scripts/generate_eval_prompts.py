#!/usr/bin/env python3
"""
Generate evaluation prompt sets from templates and datasets.

Usage:
    python scripts/generate_eval_prompts.py \
        --templates experiments/evaluation_templates/hallucination_test_set.yaml \
        --datasets experiments/datasets/pubmedqa/test.jsonl,experiments/datasets/sciq/test.jsonl \
        --output data/evaluation/test_set.jsonl \
        --samples-per-dataset 50
"""

import argparse
import json
import os
import random
import yaml
from pathlib import Path


def load_templates(path):
    """Load YAML evaluation templates."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_samples(path, num_samples, seed=42):
    """Load random samples from JSONL dataset."""
    random.seed(seed)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    
    if num_samples < len(records):
        random.shuffle(records)
        records = records[:num_samples]
    
    return records


def format_dataset_prompt(record):
    """Format a dataset record into a prompt."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    return (
        f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    ), record.get("output", "")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation prompts")
    parser.add_argument("--templates", required=True, help="YAML template file")
    parser.add_argument("--datasets", required=True, help="Comma-separated list of JSONL dataset files")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--samples-per-dataset", type=int, default=50, help="Random samples from each dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load templates
    templates = load_templates(args.templates)
    
    all_prompts = []
    
    # Add template-based prompts
    for category, domains in templates.items():
        for domain, prompts in domains.items():
            for p in prompts:
                all_prompts.append({
                    "prompt_id": f"template_{category}_{domain}_{len(all_prompts)}",
                    "source": "template",
                    "category": p.get("category", category),
                    "domain": domain,
                    "prompt": (
                        f"Below is an instruction that describes a task.\n\n"
                        f"### Instruction:\n{p['instruction']}\n\n"
                        f"### Input:\n{p.get('input', '')}\n\n"
                        f"### Response:\n"
                    ),
                    "expected_output": p.get("expected_output", ""),
                    "note": p.get("note", ""),
                })
    
    # Add dataset-based prompts
    for dataset_path in args.datasets.split(","):
        dataset_path = dataset_path.strip()
        if not os.path.exists(dataset_path):
            print(f"Warning: dataset not found: {dataset_path}")
            continue
        
        domain = os.path.basename(os.path.dirname(dataset_path))
        samples = load_dataset_samples(dataset_path, args.samples_per_dataset, args.seed)
        
        for rec in samples:
            prompt, ground_truth = format_dataset_prompt(rec)
            all_prompts.append({
                "prompt_id": f"dataset_{domain}_{len(all_prompts)}",
                "source": "dataset",
                "category": "factual",
                "domain": domain,
                "prompt": prompt,
                "expected_output": ground_truth,
                "note": "",
            })
    
    # Shuffle and save
    random.shuffle(all_prompts)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(all_prompts)} evaluation prompts")
    print(f"Saved to: {args.output}")
    
    # Print category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in all_prompts)
    domains = Counter(p["domain"] for p in all_prompts)
    print(f"\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy domain:")
    for dom, count in sorted(domains.items()):
        print(f"  {dom}: {count}")


if __name__ == "__main__":
    main()
