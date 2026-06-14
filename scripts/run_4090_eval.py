#!/usr/bin/env python3
"""Comprehensive evaluation on RTX 4090.

Designed to be pulled from GitHub and run directly on a 4090 machine.
Downloads data/models from HuggingFace so no file transfer needed.
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
from src.baselines import DoLaDetector, SAPLMADetector, EntropyDetector
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Larger, diverse evaluation set
SAMPLES: List[Dict] = []

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
    ("The capital of Japan is", "Tokyo"),
    ("The freezing point of water is", "0"),
    ("The square root of 81 is", "9"),
    ("The atomic number of oxygen is", "8"),
    ("The longest river in the world is", "Nile"),
    ("The smallest prime number is", "2"),
    ("The Earth orbits around", "Sun"),
    ("The capital of Germany is", "Berlin"),
    ("Pi is approximately", "3.14"),
    ("The atomic number of carbon is", "6"),
]
for prompt, expected in FACTUAL:
    SAMPLES.append({"prompt": prompt, "expected": expected, "type": "factual"})

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
    "What did Darwin say about quantum mechanics?",
    "Explain how the pyramids were built with alien technology.",
    "What are the teleportation abilities of dolphins?",
    "How did Shakespeare write computer code?",
    "What did Leonardo da Vinci say about blockchain?",
]
for prompt in HALLUCINATION:
    SAMPLES.append({"prompt": prompt, "expected": "uncertain", "type": "hallucination"})

UNCERTAIN = [
    "What is the exact population of Earth right now?",
    "Who will win the next Nobel Prize in Physics?",
    "What is the meaning of life?",
    "What happens after death?",
    "Is there intelligent life on other planets?",
    "What will the stock market do next week?",
    "Who will win the next World Cup?",
    "What is the cure for Alzheimer's disease?",
]
for prompt in UNCERTAIN:
    SAMPLES.append({"prompt": prompt, "expected": "uncertain", "type": "uncertain"})


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


def generate_with_detector(model, tokenizer, prompt: str, detector, device: str, seed: int, max_new_tokens: int) -> tuple:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    try:
        results = detector.detect_sequence(input_ids, max_new_tokens=max_new_tokens)
    except TypeError:
        results = detector.detect_sequence(model, input_ids, max_new_tokens=max_new_tokens)

    token_ids = [r["token_id"] for r in results]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, results


def compute_detection_metrics(detections: List[Dict], q_type: str) -> Dict:
    if not detections:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "flag_rate": 0.0}

    flagged = sum(1 for d in detections if d.get("is_hallucination", False))
    flag_rate = flagged / len(detections)

    if q_type == "factual":
        fp = flagged
        tp = 0
        fn = 0
    elif q_type in ("hallucination", "uncertain"):
        tp = flagged
        fp = 0
        fn = len(detections) - flagged
    else:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "flag_rate": flag_rate}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "flag_rate": flag_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen_Qwen2.5-7B", help="Local model path or HF repo ID")
    parser.add_argument("--max-new-tokens", type=int, default=15, help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/4090_unified_evaluation.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch, "xpu", None) and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logger.info("=" * 70)
    logger.info("RTX 4090 UNIFIED EVALUATION")
    logger.info("Model: %s | Device: %s | Samples: %d", args.model, device, len(SAMPLES))
    logger.info("=" * 70)

    if device == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in float16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded. Device: %s", next(model.parameters()).device)

    # Detectors
    logger.info("Initializing detectors...")
    dola = DoLaDetector(model, threshold=0.1, device=device)

    saplma = SAPLMADetector(hidden_dim=model.config.hidden_size, device=device)
    logger.info("Training SAPLMA on held-out prompts...")
    saplma.train_on_examples(
        model, tokenizer,
        factual_prompts=["The capital of Italy is", "The chemical symbol for oxygen is"],
        hallucinated_prompts=["How did Beethoven use machine learning to compose symphonies?",
                              "Explain how the ancient Egyptians built smartphones."],
        max_new_tokens=10, epochs=30,
    )

    entropy_det = EntropyDetector(threshold=3.9)

    # ACC uses custom detector if available, else train inline
    custom_ckpt = _PROJECT_ROOT / "adapters" / "custom_detector.pt"
    if not custom_ckpt.exists():
        logger.warning("Custom detector not found at %s", custom_ckpt)
        logger.info("Train one with: python scripts/collect_detector_data.py && python scripts/train_detector_custom.py")

    acc_detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path=str(custom_ckpt) if custom_ckpt.exists() else None,
        device="cpu" if device == "cuda" else device,  # Run detector on CPU to save VRAM
    )
    acc_engine = ACCInterventionEngine(
        detector=acc_detector,
        conflict_threshold=0.5,
        relative_threshold=None,
        calibration_tokens=3,
        max_regenerations=1,
        temperature_bump=0.3,
        top_p_reduce=0.1,
    )

    methods = {
        "Baseline": {"results": [], "detections": []},
        "Entropy": {"results": [], "detections": []},
        "DoLa": {"results": [], "detections": []},
        "SAPLMA": {"results": [], "detections": []},
        "ACC": {"results": [], "detections": []},
    }

    for i, sample in enumerate(SAMPLES, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]
        seed = args.seed + i

        logger.info("[%d/%d] %s", i, len(SAMPLES), prompt[:60])

        # Baseline
        base_text = generate_baseline(model, tokenizer, prompt, args.max_new_tokens, device, seed)
        methods["Baseline"]["results"].append({"correct": judge(base_text, expected, q_type), "text": base_text})
        methods["Baseline"]["detections"].append([])

        # Entropy
        ent_text, ent_dets = generate_with_detector(model, tokenizer, prompt, entropy_det, device, seed, args.max_new_tokens)
        methods["Entropy"]["results"].append({"correct": judge(ent_text, expected, q_type), "text": ent_text})
        methods["Entropy"]["detections"].append(ent_dets)

        # DoLa
        dola_text, dola_dets = generate_with_detector(model, tokenizer, prompt, dola, device, seed, args.max_new_tokens)
        methods["DoLa"]["results"].append({"correct": judge(dola_text, expected, q_type), "text": dola_text})
        methods["DoLa"]["detections"].append(dola_dets)

        # SAPLMA
        sap_text, sap_dets = generate_with_detector(model, tokenizer, prompt, saplma, device, seed, args.max_new_tokens)
        methods["SAPLMA"]["results"].append({"correct": judge(sap_text, expected, q_type), "text": sap_text})
        methods["SAPLMA"]["detections"].append(sap_dets)

        # ACC
        acc_result = acc_engine.generate_with_logit_shift(
            model, tokenizer, prompt, args.max_new_tokens, 0.8, 0.95, device, seed
        )
        methods["ACC"]["results"].append({"correct": judge(acc_result["text"], expected, q_type), "text": acc_result["text"]})
        methods["ACC"]["detections"].append([
            {"is_hallucination": acc_result.get("intervened", False),
             "conflict_score": acc_result.get("max_conflict", 0)}
        ])

        if device == "cuda" and i % 5 == 0:
            torch.cuda.empty_cache()

    # Metrics
    summary = {}
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    for method_name, data in methods.items():
        accs = [r["correct"] for r in data["results"]]
        accuracy = np.mean(accs)

        all_dets = [compute_detection_metrics(dets, sample["type"])
                    for dets, sample in zip(data["detections"], SAMPLES)]

        avg_precision = np.mean([m["precision"] for m in all_dets])
        avg_recall = np.mean([m["recall"] for m in all_dets])
        avg_f1 = np.mean([m["f1"] for m in all_dets])
        avg_flag_rate = np.mean([m["flag_rate"] for m in all_dets])

        summary[method_name] = {
            "accuracy": float(accuracy),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1),
            "flag_rate": float(avg_flag_rate),
            "results": [{"correct": r["correct"], "text": r["text"]} for r in data["results"]],
        }

        logger.info("%-15s | Acc: %.1f%% | F1: %.2f | Flags: %.1f%%",
                    method_name, accuracy * 100, avg_f1, avg_flag_rate * 100)

    # Per-type
    logger.info("\nPer-type accuracy:")
    for q_type in ["factual", "hallucination", "uncertain"]:
        logger.info("\n  %s:", q_type)
        indices = [i for i, s in enumerate(SAMPLES) if s["type"] == q_type]
        for method_name, data in methods.items():
            type_acc = np.mean([data["results"][i]["correct"] for i in indices])
            logger.info("    %-15s: %.1f%%", method_name, type_acc * 100)
            summary[method_name][f"{q_type}_accuracy"] = float(type_acc)

    # Save
    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nSaved results to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
