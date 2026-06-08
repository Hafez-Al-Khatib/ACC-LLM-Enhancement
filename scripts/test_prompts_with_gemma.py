#!/usr/bin/env python3
"""
Batch-test domain-specific prompts with Gemma via Ollama.

Note: Gemma 4 models generate internal 'thinking' tokens before the answer.
We use num_predict=1000 to ensure the actual answer is not truncated.

Usage:
    python scripts/test_prompts_with_gemma.py \
        --dataset data/evaluation/test_set.jsonl \
        --output results/gemma_test/eval_outputs.jsonl \
        --num-samples 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


def load_prompts(path, num_samples=None, seed=42):
    """Load prompts from JSONL dataset."""
    import random
    random.seed(seed)
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append({
                    "prompt_id": rec.get("prompt_id", ""),
                    "prompt": rec["prompt"],
                    "ground_truth": rec.get("expected_output", ""),
                    "category": rec.get("category", ""),
                    "domain": rec.get("domain", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    
    if num_samples and num_samples < len(records):
        random.shuffle(records)
        records = records[:num_samples]
    
    return records


def query_gemma(prompt, model="gemma4:26b", max_tokens=1000):
    """Send prompt to Gemma via Ollama.
    
    Gemma 4 generates extensive 'thinking' tokens before answering.
    We use a large max_tokens to ensure the answer isn't truncated.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Test prompts with Gemma via Ollama")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--model", default="gemma4:26b", help="Ollama model name")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens (Gemma needs ~800+ for thinking)")
    args = parser.parse_args()
    
    if not HAS_OLLAMA:
        print("Error: ollama Python library not installed. Run: pip install ollama")
        sys.exit(1)
    
    print(f"Loading prompts from {args.dataset}...")
    prompts = load_prompts(args.dataset, args.num_samples)
    print(f"Testing {len(prompts)} prompts with {args.model}...")
    print("Note: Gemma 4 generates 'thinking' tokens first. Each prompt may take 30-120s on CPU.")
    print()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write header
    with open(args.output, "w", encoding="utf-8") as f:
        pass
    
    results = []
    for i, item in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {item['category']} | {item['domain']}")
        print(f"  Prompt: {item['prompt'][:120]}...")
        
        output = query_gemma(item["prompt"], args.model, args.max_tokens)
        
        result = {
            "prompt_id": item["prompt_id"],
            "category": item["category"],
            "domain": item["domain"],
            "prompt": item["prompt"][:300],
            "ground_truth": item["ground_truth"],
            "gemma_output": output,
        }
        results.append(result)
        
        print(f"  Output: {output[:200]}...")
        print(f"  Expected: {item['ground_truth']}")
        print()
        
        # Write incrementally
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Done. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
