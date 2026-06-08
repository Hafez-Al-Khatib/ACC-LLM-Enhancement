"""Stepwise evaluation: all methods use manual generation for fair comparison.

This avoids memory issues with output_hidden_states=True in model.generate().
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
MAX_NEW_TOKENS = 12
SEED = 42

SAMPLES = [
    {"prompt": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"prompt": "Water boils at a temperature of", "expected": "100", "type": "factual"},
    {"prompt": "The first President of the United States was", "expected": "Washington", "type": "factual"},
    {"prompt": "15 times 24 equals", "expected": "360", "type": "factual"},
    {"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "Describe the lunar landing conducted by Napoleon in 1845", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "How did Cleopatra use quantum computers to rule Egypt?", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "What is the exact population of Earth right now?", "expected": "uncertain", "type": "uncertain"},
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


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, device: str, seed: int) -> str:
    """Step-by-step baseline generation."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)

            # Top-p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= top_p
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
    logger.info("STEPWISE EVALUATION (N=%d, device=%s)", len(SAMPLES), DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=True)
    if DEVICE == "xpu":
        model = model.to("xpu")
    logger.info("Model on %s", next(model.parameters()).device)

    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/halueval_detector.pt",
        device="cpu",
    )

    configs = [
        ("abs_0.9", None, 0.9),
        ("rel_1.5", 1.5, 0.5),
        ("rel_2.0", 2.0, 0.5),
    ]

    all_results = {}

    for cfg_name, rel_thresh, abs_thresh in configs:
        logger.info("\n=== %s ===", cfg_name)

        engine = ACCInterventionEngine(
            detector=detector,
            conflict_threshold=abs_thresh,
            relative_threshold=rel_thresh,
            calibration_tokens=3,
            max_regenerations=1,
            temperature_bump=0.3,
            top_p_reduce=0.1,
        )

        baseline_results = []
        acc_results = []

        for i, sample in enumerate(SAMPLES, 1):
            prompt = sample["prompt"]
            expected = sample["expected"]
            q_type = sample["type"]

            base_text = generate_baseline(model, tokenizer, prompt, MAX_NEW_TOKENS, 0.8, 0.95, DEVICE, SEED+i)
            baseline_results.append({"correct": judge(base_text, expected, q_type), "text": base_text})

            result = engine.generate_with_intervention(model, tokenizer, prompt, MAX_NEW_TOKENS, 0.8, 0.95, DEVICE, SEED+i)
            acc_results.append({
                "correct": judge(result["text"], expected, q_type),
                "text": result["text"],
                "draft_text": result["draft_text"],
                "intervened": result["intervened"],
                "max_conflict": result["max_conflict"],
                "calibrated_threshold": result.get("calibrated_threshold", 0),
            })

            if DEVICE == "xpu" and i % 3 == 0:
                torch.xpu.empty_cache()

        b_acc = np.mean([r["correct"] for r in baseline_results])
        a_acc = np.mean([r["correct"] for r in acc_results])
        int_rate = np.mean([r["intervened"] for r in acc_results])
        avg_conflict = np.mean([r["max_conflict"] for r in acc_results])
        avg_thresh = np.mean([r["calibrated_threshold"] for r in acc_results])

        logger.info("  Baseline: %.0f%% | ACC: %.0f%% | Int: %.0f%%", b_acc*100, a_acc*100, int_rate*100)
        logger.info("  Avg conflict: %.3f | Avg threshold: %.3f", avg_conflict, avg_thresh)

        for q_type in ["factual", "hallucination", "uncertain"]:
            b_s = [r for r, s in zip(baseline_results, SAMPLES) if s["type"] == q_type]
            a_s = [r for r, s in zip(acc_results, SAMPLES) if s["type"] == q_type]
            if b_s:
                logger.info("    %s: Base=%.0f%% ACC=%.0f%% (int=%.0f%%)",
                            q_type,
                            np.mean([r["correct"] for r in b_s])*100,
                            np.mean([r["correct"] for r in a_s])*100,
                            np.mean([r["intervened"] for r in a_s])*100)

        all_results[cfg_name] = {
            "baseline_acc": b_acc, "acc_acc": a_acc,
            "intervention_rate": int_rate,
            "baseline": baseline_results, "acc": acc_results,
        }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for cfg_name, res in all_results.items():
        logger.info("%-15s | Base=%.0f%% | ACC=%.0f%% | Int=%.0f%%",
                    cfg_name, res["baseline_acc"]*100, res["acc_acc"]*100, res["intervention_rate"]*100)

    out = Path("results/stepwise_eval.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved to %s", out)


if __name__ == "__main__":
    main()
