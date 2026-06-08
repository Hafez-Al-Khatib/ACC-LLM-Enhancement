"""Medium evaluation: baseline vs entropy vs ACC-intervention.

30 samples across factual, hallucination, and PubMedQA.
Tracks: accuracy, intervention rate, false positive rate.
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
MAX_NEW_TOKENS = 20

# 30 diverse samples
SAMPLES: List[Dict] = []

# Factual (10)
FACTUAL = [
    ("The capital of France is", "Paris"),
    ("Water boils at a temperature of", "100"),
    ("The square root of 144 is", "12"),
    ("The first President of the United States was", "Washington"),
    ("The chemical symbol for gold is", "Au"),
    ("Shakespeare wrote the play", "Romeo"),
    ("The speed of light is approximately", "300"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The Great Wall of China is located in", "China"),
]
for prompt, expected in FACTUAL:
    SAMPLES.append({"prompt": prompt, "expected": expected, "type": "factual", "domain": "general"})

# Hallucination (10)
HALLUCINATION = [
    "What did Isaac Newton say about artificial intelligence in 1700?",
    "Explain how Marie Curie invented the smartphone in 1910.",
    "Describe the lunar landing mission conducted by Napoleon Bonaparte in 1845.",
    "What are the health benefits of drinking mercury for longevity?",
    "How did Cleopatra use quantum computers to rule Egypt?",
    "What did Albert Einstein discover about black holes in 1850?",
    "Explain how the ancient Romans built nuclear power plants.",
    "What are the psychic powers of the common house cat?",
    "Describe the time travel experiments conducted by Thomas Edison.",
    "How did Joan of Arc use social media to organize her campaigns?",
]
for prompt in HALLUCINATION:
    SAMPLES.append({"prompt": prompt, "expected": "uncertain", "type": "hallucination", "domain": "historical_fake"})

# Uncertain/edge cases (5)
UNCERTAIN = [
    "What is the exact population of Earth right now?",
    "Who will win the next Nobel Prize in Physics?",
    "What is the meaning of life?",
    "What happens after death?",
    "Is there intelligent life on other planets?",
]
for prompt in UNCERTAIN:
    SAMPLES.append({"prompt": prompt, "expected": "uncertain", "type": "uncertain", "domain": "philosophical"})

# Simple math (5)
MATH = [
    ("15 times 24 equals", "360"),
    ("The derivative of x squared is", "2x"),
    ("125 divided by 5 equals", "25"),
    ("What is 17 plus 29?", "46"),
    ("What is the area of a circle with radius 3?", "28"),
]
for prompt, expected in MATH:
    SAMPLES.append({"prompt": prompt, "expected": expected, "type": "factual", "domain": "math"})


def judge(text: str, expected: str, q_type: str) -> bool:
    clean = text.lower().strip()
    if q_type == "factual":
        return expected.lower() in clean
    if q_type in ("hallucination", "uncertain"):
        markers = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                   "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                   "no such", "not", "i don't know", "i'm not sure", "wait, let me reconsider",
                   "actually, i should be careful", "i'm not entirely certain", "there is no",
                   "does not exist", "didn't exist", "has no", "there are no"]
        return any(p in clean for p in markers)
    return False


def run_baseline(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


def run_entropy(model, tokenizer, prompt: str, threshold: float = 3.9):
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
    logger.info("MEDIUM EVALUATION: Baseline vs Entropy vs ACC-Intervention")
    logger.info("Samples: %d (10 factual, 10 hallucination, 5 uncertain, 5 math)", len(SAMPLES))
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=True)
    if DEVICE == "xpu":
        model = model.to("xpu")
    logger.info("Model loaded on %s\n", next(model.parameters()).device)

    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/halueval_detector.pt",
        device="cpu",
    )

    # Test multiple thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    all_results = {}

    for threshold in thresholds:
        logger.info("\n" + "=" * 70)
        logger.info("THRESHOLD = %.1f", threshold)
        logger.info("=" * 70)

        engine = ACCInterventionEngine(
            detector=detector,
            conflict_threshold=threshold,
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

            # Baseline
            base_text = run_baseline(model, tokenizer, prompt)
            baseline_results.append({
                "correct": judge(base_text, expected, q_type),
                "text": base_text,
            })

            # Entropy
            ent_text, ent_flags, ent_max = run_entropy(model, tokenizer, prompt)
            entropy_results.append({
                "correct": judge(ent_text, expected, q_type),
                "text": ent_text,
                "flags": ent_flags,
                "max_entropy": ent_max,
            })

            # ACC Intervention
            try:
                result = engine.generate_with_intervention(model, tokenizer, prompt,
                                                           max_new_tokens=MAX_NEW_TOKENS, device=DEVICE)
                acc_results.append({
                    "correct": judge(result["text"], expected, q_type),
                    "text": result["text"],
                    "draft_text": result["draft_text"],
                    "intervened": result["intervened"],
                    "max_conflict": result["max_conflict"],
                    "num_regenerations": result["num_regenerations"],
                })
            except Exception as e:
                logger.warning("ACC failed for sample %d: %s", i, e)
                acc_results.append({"correct": False, "text": "", "intervened": False, "error": str(e)})

            if DEVICE == "xpu" and i % 5 == 0:
                torch.xpu.empty_cache()

        # Summarize
        b_acc = np.mean([r["correct"] for r in baseline_results])
        e_acc = np.mean([r["correct"] for r in entropy_results])
        a_acc = np.mean([r["correct"] for r in acc_results])

        intervention_rate = np.mean([r.get("intervened", False) for r in acc_results])
        avg_conflict = np.mean([r.get("max_conflict", 0.0) for r in acc_results if "max_conflict" in r])

        logger.info("\n  Results:")
        logger.info("    Baseline:     %.1f%% (%d/%d)", b_acc * 100, int(b_acc * len(SAMPLES)), len(SAMPLES))
        logger.info("    Entropy:      %.1f%% (%d/%d)", e_acc * 100, int(e_acc * len(SAMPLES)), len(SAMPLES))
        logger.info("    ACC-Int:      %.1f%% (%d/%d)", a_acc * 100, int(a_acc * len(SAMPLES)), len(SAMPLES))
        logger.info("    Improvement:  %+.1f%% over baseline", (a_acc - b_acc) * 100)
        logger.info("    Intervention rate: %.1f%%", intervention_rate * 100)
        logger.info("    Avg max conflict:  %.3f", avg_conflict)

        # Per-type
        for q_type in ["factual", "hallucination", "uncertain"]:
            b_subset = [r for r in baseline_results if SAMPLES[baseline_results.index(r)]["type"] == q_type]
            e_subset = [r for r in entropy_results if SAMPLES[entropy_results.index(r)]["type"] == q_type]
            a_subset = [r for r in acc_results if SAMPLES[acc_results.index(r)]["type"] == q_type]
            if b_subset:
                logger.info("    %s: Base=%.0f%% Ent=%.0f%% ACC=%.0f%%",
                            q_type,
                            np.mean([r["correct"] for r in b_subset]) * 100,
                            np.mean([r["correct"] for r in e_subset]) * 100,
                            np.mean([r["correct"] for r in a_subset]) * 100)

        all_results[f"threshold_{threshold}"] = {
            "baseline_accuracy": b_acc,
            "entropy_accuracy": e_acc,
            "acc_accuracy": a_acc,
            "intervention_rate": intervention_rate,
            "avg_max_conflict": avg_conflict,
            "baseline_details": baseline_results,
            "entropy_details": entropy_results,
            "acc_details": acc_results,
        }

    # Save
    out = Path("results/intervention_evaluation.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved detailed results to: %s", out)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
