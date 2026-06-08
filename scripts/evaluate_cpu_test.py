"""CPU test: 5 samples, verify intervention logic works."""

from __future__ import annotations

import json
import logging
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "cpu"  # Force CPU
MAX_NEW_TOKENS = 12
SEED = 42

SAMPLES = [
    {"prompt": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "15 times 24 equals", "expected": "360", "type": "factual"},
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


def main():
    logger.info("CPU TEST: 5 samples")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                                  local_files_only=True, trust_remote_code=True)
    model = model.to(DEVICE)
    logger.info("Model on CPU")

    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/halueval_detector.pt",
        device="cpu",
    )

    engine = ACCInterventionEngine(
        detector=detector,
        conflict_threshold=0.7,
        relative_threshold=None,
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

        base_text = engine.generate_simple_baseline(model, tokenizer, prompt, MAX_NEW_TOKENS, device=DEVICE, seed=SEED+i)
        baseline_results.append({"correct": judge(base_text, expected, q_type), "text": base_text})

        result = engine.generate_with_intervention(model, tokenizer, prompt, MAX_NEW_TOKENS, device=DEVICE, seed=SEED+i)
        acc_results.append({
            "correct": judge(result["text"], expected, q_type),
            "text": result["text"],
            "draft_text": result["draft_text"],
            "intervened": result["intervened"],
            "max_conflict": result["max_conflict"],
        })

        logger.info("\n%d. %s [%s]", i, prompt[:50], q_type)
        logger.info("   Base: %s [%s]", base_text[:80], "✓" if baseline_results[-1]["correct"] else "✗")
        logger.info("   ACC:  %s [%s] (int=%s, conflict=%.2f)",
                    result["text"][:80], "✓" if acc_results[-1]["correct"] else "✗",
                    result["intervened"], result["max_conflict"])

    b_acc = np.mean([r["correct"] for r in baseline_results])
    a_acc = np.mean([r["correct"] for r in acc_results])
    int_rate = np.mean([r["intervened"] for r in acc_results])

    logger.info("\nBaseline: %.0f%% | ACC: %.0f%% | Intervention: %.0f%%", b_acc*100, a_acc*100, int_rate*100)

    out = Path("results/cpu_test.json")
    with open(out, "w") as f:
        json.dump({"baseline": baseline_results, "acc": acc_results}, f, indent=2)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
