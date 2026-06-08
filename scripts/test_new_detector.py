"""Quick test: new detector with relative thresholding."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "cpu"
MAX_NEW_TOKENS = 12
SEED = 42

SAMPLES = [
    {"prompt": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"prompt": "15 times 24 equals", "expected": "360", "type": "factual"},
    {"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "Who will win the next Nobel Prize in Physics?", "expected": "uncertain", "type": "uncertain"},
]


def judge(text: str, expected: str, q_type: str) -> bool:
    clean = text.lower().strip()
    if q_type == "factual":
        return expected.lower() in clean
    if q_type in ("hallucination", "uncertain"):
        markers = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                   "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                   "no such", "not", "i don't know", "i'm not sure", "wait, let me reconsider",
                   "actually, i should be careful", "i'm not entirely certain", "there is no",
                   "does not exist", "didn't exist", "has no", "there are no", "as an ai"]
        return any(p in clean for p in markers)
    return False


def generate_baseline(model, tokenizer, prompt: str, device: str, seed: int) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= 0.95
            mask[1:] = mask[:-1].clone()
            mask[0] = True
            filtered_probs = sorted_probs * mask.to(sorted_probs.dtype)
            filtered_probs = filtered_probs / filtered_probs.sum()
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, filtered_probs)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    print("TESTING NEW DETECTOR (5 samples, CPU)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                                  local_files_only=True, trust_remote_code=True)
    model = model.to(DEVICE)

    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/custom_detector.pt",
        device="cpu",
    )

    engine = ACCInterventionEngine(
        detector=detector, conflict_threshold=0.5, relative_threshold=1.5,
        calibration_tokens=3, max_regenerations=1, temperature_bump=0.3, top_p_reduce=0.1,
    )

    baseline_results = []
    acc_results = []

    for i, sample in enumerate(SAMPLES, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]

        base_text = generate_baseline(model, tokenizer, prompt, DEVICE, SEED+i)
        baseline_results.append({"correct": judge(base_text, expected, q_type), "text": base_text})

        result = engine.generate_with_intervention(model, tokenizer, prompt, MAX_NEW_TOKENS, 0.8, 0.95, DEVICE, SEED+i)
        acc_results.append({
            "correct": judge(result["text"], expected, q_type),
            "text": result["text"],
            "intervened": result["intervened"],
            "max_conflict": result["max_conflict"],
            "threshold": result.get("calibrated_threshold", 0),
        })

    b_acc = np.mean([r["correct"] for r in baseline_results])
    a_acc = np.mean([r["correct"] for r in acc_results])
    int_rate = np.mean([r["intervened"] for r in acc_results])

    print(f"\nBaseline: {b_acc*100:.0f}% | ACC: {a_acc*100:.0f}% | Int: {int_rate*100:.0f}%")
    for i, (b, a) in enumerate(zip(baseline_results, acc_results), 1):
        print(f"\n{i}. {SAMPLES[i-1]['prompt'][:40]} [{SAMPLES[i-1]['type']}]")
        print(f"   Base: {b['text'][:60]} [{'OK' if b['correct'] else 'FAIL'}]")
        print(f"   ACC:  {a['text'][:60]} [{'OK' if a['correct'] else 'FAIL'}] (int={a['intervened']}, conf={a['max_conflict']:.2f})")

    with open("results/new_detector_test.json", "w") as f:
        json.dump({"baseline": baseline_results, "acc": acc_results}, f, indent=2)
    print("\nSaved to results/new_detector_test.json")


if __name__ == "__main__":
    main()
