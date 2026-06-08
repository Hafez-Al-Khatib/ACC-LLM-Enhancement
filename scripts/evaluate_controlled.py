"""Controlled evaluation: same random seed, per-prompt calibration.

Compares baseline, entropy-only, and ACC-intervention fairly.
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
MAX_NEW_TOKENS = 15
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


def run_baseline(model, tokenizer, prompt: str, seed: int) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


def run_entropy(model, tokenizer, prompt: str, threshold: float, seed: int):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95, return_dict_in_generate=True,
                             output_scores=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out.sequences[0, input_len:], skip_special_tokens=True)
    entropies = []
    for s in out.scores:
        probs = F.softmax(s[0], dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropies.append((-(probs * log_probs).sum()).item())
    flags = sum(1 for h in entropies if h > threshold)
    return text, flags, max(entropies) if entropies else 0.0


def main():
    logger.info("=" * 70)
    logger.info("CONTROLLED EVALUATION (same seed=%d, N=%d)", SEED, len(SAMPLES))
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=True)
    if DEVICE == "xpu":
        model = model.to("xpu")
    logger.info("Model on %s\n", next(model.parameters()).device)

    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/halueval_detector.pt",
        device="cpu",
    )

    configs = [
        ("absolute_0.5", {"conflict_threshold": 0.5, "relative_threshold": None}),
        ("absolute_0.7", {"conflict_threshold": 0.7, "relative_threshold": None}),
        ("absolute_0.9", {"conflict_threshold": 0.9, "relative_threshold": None}),
        ("relative_1.5", {"conflict_threshold": 0.5, "relative_threshold": 1.5}),
        ("relative_2.0", {"conflict_threshold": 0.5, "relative_threshold": 2.0}),
    ]

    all_results = {}

    for cfg_name, cfg in configs:
        logger.info("\n--- Config: %s ---", cfg_name)

        engine = ACCInterventionEngine(
            detector=detector,
            conflict_threshold=cfg["conflict_threshold"],
            relative_threshold=cfg["relative_threshold"],
            calibration_tokens=3,
            max_regenerations=1,
            temperature_bump=0.3,
            top_p_reduce=0.1,
        )

        baseline_results = []
        entropy_results = []
        acc_results = []

        for i, sample in enumerate(SAMPLES, 1):
            prompt = sample["prompt"]
            expected = sample["expected"]
            q_type = sample["type"]

            base_text = run_baseline(model, tokenizer, prompt, SEED + i)
            baseline_results.append({"correct": judge(base_text, expected, q_type), "text": base_text})

            ent_text, ent_flags, ent_max = run_entropy(model, tokenizer, prompt, 3.9, SEED + i)
            entropy_results.append({"correct": judge(ent_text, expected, q_type), "text": ent_text,
                                    "flags": ent_flags, "max_entropy": ent_max})

            try:
                result = engine.generate_with_intervention(
                    model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, device=DEVICE, seed=SEED + i
                )
                acc_results.append({
                    "correct": judge(result["text"], expected, q_type),
                    "text": result["text"],
                    "draft_text": result["draft_text"],
                    "intervened": result["intervened"],
                    "max_conflict": result["max_conflict"],
                    "calibrated_threshold": result["calibrated_threshold"],
                })
            except Exception as e:
                logger.warning("ACC failed: %s", e)
                acc_results.append({"correct": False, "error": str(e)})

            if DEVICE == "xpu" and i % 3 == 0:
                torch.xpu.empty_cache()

        b_acc = np.mean([r["correct"] for r in baseline_results])
        e_acc = np.mean([r["correct"] for r in entropy_results])
        a_acc = np.mean([r["correct"] for r in acc_results])
        int_rate = np.mean([r.get("intervened", False) for r in acc_results])

        logger.info("  Baseline: %.0f%% | Entropy: %.0f%% | ACC: %.0f%% (int=%.0f%%)",
                    b_acc*100, e_acc*100, a_acc*100, int_rate*100)

        # Per-type
        for q_type in ["factual", "hallucination", "uncertain"]:
            b_s = [r for r, s in zip(baseline_results, SAMPLES) if s["type"] == q_type]
            e_s = [r for r, s in zip(entropy_results, SAMPLES) if s["type"] == q_type]
            a_s = [r for r, s in zip(acc_results, SAMPLES) if s["type"] == q_type]
            if b_s:
                logger.info("    %s: Base=%.0f%% Ent=%.0f%% ACC=%.0f%%",
                            q_type,
                            np.mean([r["correct"] for r in b_s])*100,
                            np.mean([r["correct"] for r in e_s])*100,
                            np.mean([r["correct"] for r in a_s])*100)

        all_results[cfg_name] = {
            "baseline_acc": b_acc, "entropy_acc": e_acc, "acc_acc": a_acc,
            "intervention_rate": int_rate,
            "baseline": baseline_results, "entropy": entropy_results, "acc": acc_results,
        }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY ACROSS CONFIGURATIONS")
    logger.info("=" * 70)
    for cfg_name, res in all_results.items():
        logger.info("%-20s | Base=%.0f%% | Ent=%.0f%% | ACC=%.0f%% | Int=%.0f%%",
                    cfg_name, res["baseline_acc"]*100, res["entropy_acc"]*100,
                    res["acc_acc"]*100, res["intervention_rate"]*100)

    out = Path("results/controlled_eval.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved to %s", out)


if __name__ == "__main__":
    main()
