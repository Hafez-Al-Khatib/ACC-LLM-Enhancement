#!/usr/bin/env python3
"""Ablation study for ACC intervention components.

Tests the effect of:
1. Detector checkpoint (HaluEval vs custom)
2. Intervention strategy (none, phrase, logit-shift)
3. Threshold strategy (absolute vs relative calibration)
4. Temperature bump during regeneration
"""

from __future__ import annotations

import argparse
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


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_config(model, tokenizer, detector, config: Dict, samples: List[Dict], device: str, seed: int) -> Dict:
    """Run one ablation configuration."""
    engine = ACCInterventionEngine(
        detector=detector,
        conflict_threshold=config["threshold"],
        relative_threshold=config["relative_threshold"],
        calibration_tokens=3,
        max_regenerations=1,
        temperature_bump=config["temperature_bump"],
        top_p_reduce=0.1,
        uncertainty_bias=config.get("uncertainty_bias", 2.0),
    )

    results = []
    for i, sample in enumerate(samples, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]

        if config["intervention"] == "none":
            text = generate_baseline(model, tokenizer, prompt, config["max_new_tokens"], device, seed + i)
        elif config["intervention"] == "phrase":
            result = engine.generate_with_intervention(
                model, tokenizer, prompt, config["max_new_tokens"], 0.8, 0.95, device, seed + i
            )
            text = result["text"]
        elif config["intervention"] == "logit-shift":
            result = engine.generate_with_logit_shift(
                model, tokenizer, prompt, config["max_new_tokens"], 0.8, 0.95, device, seed + i
            )
            text = result["text"]
        else:
            raise ValueError(f"Unknown intervention: {config['intervention']}")

        results.append({
            "correct": judge(text, expected, q_type),
            "text": text,
            "type": q_type,
        })

    accuracy = np.mean([r["correct"] for r in results])
    return {"accuracy": float(accuracy), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen2.5-1.5b")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/ablation_study.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch, "xpu", None) and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logger.info("=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("Model: %s | Device: %s | Samples: %d", args.model, device, len(SAMPLES))
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded")

    # Load detectors
    detectors = {}
    halueval_ckpt = _PROJECT_ROOT / "adapters" / "halueval_detector.pt"
    custom_ckpt = _PROJECT_ROOT / "adapters" / "custom_detector.pt"

    if halueval_ckpt.exists():
        detectors["HaluEval"] = HaluEvalDetector(
            hidden_dim=model.config.hidden_size,
            checkpoint_path=str(halueval_ckpt),
            device="cpu" if device == "cuda" else device,
        )
    else:
        logger.warning("HaluEval detector not found at %s", halueval_ckpt)

    if custom_ckpt.exists():
        detectors["Custom"] = HaluEvalDetector(
            hidden_dim=model.config.hidden_size,
            checkpoint_path=str(custom_ckpt),
            device="cpu" if device == "cuda" else device,
        )
    else:
        logger.warning("Custom detector not found at %s", custom_ckpt)

    if not detectors:
        logger.error("No detectors found. Train one first.")
        return

    # Ablation configurations
    configs = []

    # Baseline: no detector
    configs.append({
        "name": "Baseline (no intervention)",
        "detector": None,
        "intervention": "none",
        "threshold": 0.5,
        "relative_threshold": None,
        "temperature_bump": 0.0,
        "max_new_tokens": args.max_new_tokens,
    })

    for detector_name in detectors.keys():
        # Phrase intervention, absolute threshold
        configs.append({
            "name": f"{detector_name} + phrase (abs 0.5)",
            "detector": detector_name,
            "intervention": "phrase",
            "threshold": 0.5,
            "relative_threshold": None,
            "temperature_bump": 0.3,
            "max_new_tokens": args.max_new_tokens,
        })

        # Phrase intervention, relative threshold
        configs.append({
            "name": f"{detector_name} + phrase (rel 1.5)",
            "detector": detector_name,
            "intervention": "phrase",
            "threshold": 0.5,
            "relative_threshold": 1.5,
            "temperature_bump": 0.3,
            "max_new_tokens": args.max_new_tokens,
        })

        # Logit shift, absolute threshold
        configs.append({
            "name": f"{detector_name} + logit-shift (abs 0.5)",
            "detector": detector_name,
            "intervention": "logit-shift",
            "threshold": 0.5,
            "relative_threshold": None,
            "temperature_bump": 0.0,
            "uncertainty_bias": 2.0,
            "max_new_tokens": args.max_new_tokens,
        })

        # Logit shift, relative threshold
        configs.append({
            "name": f"{detector_name} + logit-shift (rel 1.5)",
            "detector": detector_name,
            "intervention": "logit-shift",
            "threshold": 0.5,
            "relative_threshold": 1.5,
            "temperature_bump": 0.0,
            "uncertainty_bias": 2.0,
            "max_new_tokens": args.max_new_tokens,
        })

    # Run ablations
    summary = {}
    logger.info("\nRunning ablations...")
    for config in configs:
        logger.info("\n--- %s ---", config["name"])
        detector = detectors.get(config["detector"]) if config["detector"] else None
        result = run_config(model, tokenizer, detector, config, SAMPLES, device, args.seed)
        summary[config["name"]] = result
        logger.info("  Accuracy: %.1f%%", result["accuracy"] * 100)

        if device == "cuda":
            torch.cuda.empty_cache()

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    for name, result in summary.items():
        logger.info("%-45s | %.1f%%", name, result["accuracy"] * 100)

    # Save
    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": {k: {"accuracy": v["accuracy"]} for k, v in summary.items()},
            "detailed_results": summary,
        }, f, indent=2)
    logger.info("\nSaved to: %s", out_path)


if __name__ == "__main__":
    main()
