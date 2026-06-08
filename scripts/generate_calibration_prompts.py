#!/usr/bin/env python3
"""
Generate calibration prompts for entropy threshold estimation.

Extracts factual prompts from training datasets and formats them
for use with EntropyMonitor.calibrate().

Usage:
    python scripts/generate_calibration_prompts.py \
        --dataset experiments/datasets/pubmedqa/train.jsonl \
        --output data/calibration/pubmedqa_cal.jsonl \
        --num-prompts 100

    python scripts/generate_calibration_prompts.py \
        --dataset experiments/datasets/sciq/train.jsonl \
        --output data/calibration/sciq_cal.jsonl \
        --num-prompts 100
"""

import argparse
import json
import os
import random
from pathlib import Path


def load_dataset(path):
    """Load JSONL dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def format_prompt(record, style="alpaca"):
    """Format a dataset record into a prompt for calibration."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    
    if style == "alpaca":
        if input_text:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    elif style == "plain":
        prompt = instruction
        if input_text:
            prompt += f"\n{input_text}"
        prompt += "\nAnswer: "
    elif style == "chatml":
        prompt = f"<|im_start|>user\n{instruction}"
        if input_text:
            prompt += f"\n{input_text}"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = instruction
        if input_text:
            prompt += f" {input_text}"
    
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate calibration prompts from datasets")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to extract")
    parser.add_argument("--style", default="alpaca", choices=["alpaca", "plain", "chatml"], help="Prompt formatting style")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    records = load_dataset(args.dataset)
    if not records:
        print(f"Error: no records loaded from {args.dataset}")
        return
    
    # Shuffle and select
    random.shuffle(records)
    selected = records[:args.num_prompts]
    
    # Format prompts
    outputs = []
    for rec in selected:
        prompt = format_prompt(rec, style=args.style)
        outputs.append({
            "prompt": prompt,
            "ground_truth": rec.get("output", rec.get("answer", "")),
            "source_dataset": os.path.basename(args.dataset),
        })
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(outputs)} calibration prompts")
    print(f"Saved to: {args.output}")
    print(f"\nSample prompt:\n{'-'*40}")
    print(outputs[0]["prompt"][:300] + "..." if len(outputs[0]["prompt"]) > 300 else outputs[0]["prompt"])


if __name__ == "__main__":
    main()
