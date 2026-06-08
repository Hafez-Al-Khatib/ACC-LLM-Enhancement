"""Unified evaluation: Baseline, Entropy, DoLa, SAPLMA, ACC.

Fair comparison: same prompts, same seeds, same judge function.
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
from src.baselines import DoLaDetector, SAPLMADetector, EntropyDetector
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def generate_baseline(model, tokenizer, prompt: str, device: str, seed: int) -> str:
    """Generate baseline text using pure softmax sampling (no top-p).
    Matches the sampling strategy used by detector-based methods."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_with_detector(model, tokenizer, prompt: str, detector, device: str, seed: int) -> tuple:
    """Generate text using a detector's detect_sequence method.
    Returns (text, list of detection results).
    """
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Some detectors need model passed explicitly
    try:
        results = detector.detect_sequence(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    except TypeError:
        results = detector.detect_sequence(model, input_ids, max_new_tokens=MAX_NEW_TOKENS)

    # Reconstruct text from token ids
    token_ids = [r["token_id"] for r in results]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    return text, results


def compute_detection_metrics(detections: List[Dict], q_type: str) -> Dict:
    """Compute precision/recall of hallucination detection.

    For factual prompts: hallucination flag = false positive
    For hallucination prompts: no flag = false negative
    """
    if not detections:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "flag_rate": 0.0}

    flagged = sum(1 for d in detections if d.get("is_hallucination", False))
    flag_rate = flagged / len(detections)

    if q_type == "factual":
        # False positives = flagged tokens on factual prompts
        fp = flagged
        tp = 0
        fn = 0
    elif q_type in ("hallucination", "uncertain"):
        # True positives = flagged tokens on hallucination prompts
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
    logger.info("=" * 70)
    logger.info("UNIFIED EVALUATION: All Methods")
    logger.info("Samples: %d | Device: %s | Max tokens: %d", len(SAMPLES), DEVICE, MAX_NEW_TOKENS)
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                                  local_files_only=True, trust_remote_code=True)
    model = model.to(DEVICE)
    logger.info("Model loaded on %s\n", DEVICE)

    # Initialize detectors
    logger.info("Initializing detectors...")

    # DoLa
    dola = DoLaDetector(model, threshold=0.1, device=DEVICE)

    # SAPLMA (train inline) — use held-out prompts NOT in evaluation set
    saplma = SAPLMADetector(hidden_dim=model.config.hidden_size, device=DEVICE)
    logger.info("Training SAPLMA...")
    factual_train = [
        "The capital of Italy is",
        "The chemical symbol for oxygen is",
    ]
    halluc_train = [
        "How did Beethoven use machine learning to compose symphonies?",
        "Explain how the ancient Egyptians built smartphones.",
    ]
    saplma.train_on_examples(model, tokenizer, factual_train, halluc_train, max_new_tokens=8, epochs=20)

    # Entropy
    entropy_det = EntropyDetector(threshold=3.9)

    # ACC (custom detector)
    acc_detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path="adapters/custom_detector.pt",
        device="cpu",
    )
    acc_engine = ACCInterventionEngine(
        detector=acc_detector, conflict_threshold=0.5, relative_threshold=1.5,
        calibration_tokens=3, max_regenerations=1, temperature_bump=0.3, top_p_reduce=0.1,
    )

    # Results storage
    methods = {
        "Baseline": {"results": [], "detections": []},
        "Entropy": {"results": [], "detections": []},
        "DoLa": {"results": [], "detections": []},
        "SAPLMA": {"results": [], "detections": []},
        "ACC": {"results": [], "detections": []},
    }

    # Run evaluation
    for i, sample in enumerate(SAMPLES, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]
        seed = SEED + i

        logger.info("Sample %d/%d: %s", i, len(SAMPLES), prompt[:50])

        # Baseline
        base_text = generate_baseline(model, tokenizer, prompt, DEVICE, seed)
        methods["Baseline"]["results"].append({"correct": judge(base_text, expected, q_type), "text": base_text})
        methods["Baseline"]["detections"].append([])

        # Entropy
        ent_text, ent_dets = generate_with_detector(model, tokenizer, prompt, entropy_det, DEVICE, seed)
        methods["Entropy"]["results"].append({"correct": judge(ent_text, expected, q_type), "text": ent_text})
        methods["Entropy"]["detections"].append(ent_dets)

        # DoLa
        dola_text, dola_dets = generate_with_detector(model, tokenizer, prompt, dola, DEVICE, seed)
        methods["DoLa"]["results"].append({"correct": judge(dola_text, expected, q_type), "text": dola_text})
        methods["DoLa"]["detections"].append(dola_dets)

        # SAPLMA
        sap_text, sap_dets = generate_with_detector(model, tokenizer, prompt, saplma, DEVICE, seed)
        methods["SAPLMA"]["results"].append({"correct": judge(sap_text, expected, q_type), "text": sap_text})
        methods["SAPLMA"]["detections"].append(sap_dets)

        # ACC
        acc_result = acc_engine.generate_with_intervention(model, tokenizer, prompt, MAX_NEW_TOKENS, 0.8, 0.95, DEVICE, seed)
        methods["ACC"]["results"].append({"correct": judge(acc_result["text"], expected, q_type), "text": acc_result["text"]})
        # Create pseudo-detections from ACC conflict scores
        acc_dets = [{"is_hallucination": acc_result.get("intervened", False), "conflict_score": acc_result.get("max_conflict", 0)}]
        methods["ACC"]["detections"].append(acc_dets)

    # Compute metrics
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    summary = {}
    for method_name, data in methods.items():
        accs = [r["correct"] for r in data["results"]]
        accuracy = np.mean(accs)

        # Detection metrics
        all_dets = []
        for dets, sample in zip(data["detections"], SAMPLES):
            metrics = compute_detection_metrics(dets, sample["type"])
            all_dets.append(metrics)

        avg_precision = np.mean([m["precision"] for m in all_dets])
        avg_recall = np.mean([m["recall"] for m in all_dets])
        avg_f1 = np.mean([m["f1"] for m in all_dets])
        avg_flag_rate = np.mean([m["flag_rate"] for m in all_dets])

        summary[method_name] = {
            "accuracy": accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "flag_rate": avg_flag_rate,
            "results": data["results"],
            "detections": data["detections"],
        }

        logger.info("%-15s | Acc: %.0f%% | Prec: %.2f | Rec: %.2f | F1: %.2f | Flags: %.1f%%",
                    method_name, accuracy * 100, avg_precision, avg_recall, avg_f1, avg_flag_rate * 100)

    # Per-type breakdown
    logger.info("\nPer-type accuracy:")
    for q_type in ["factual", "hallucination", "uncertain"]:
        logger.info("\n  %s:", q_type)
        for method_name, data in methods.items():
            indices = [i for i, s in enumerate(SAMPLES) if s["type"] == q_type]
            if indices:
                type_acc = np.mean([data["results"][i]["correct"] for i in indices])
                logger.info("    %-15s: %.0f%%", method_name, type_acc * 100)

    # Save
    out = Path("results/unified_evaluation.json")
    with open(out, "w") as f:
        # Convert numpy types for JSON serialization
        json_summary = {}
        for k, v in summary.items():
            json_summary[k] = {
                "accuracy": float(v["accuracy"]),
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1": float(v["f1"]),
                "flag_rate": float(v["flag_rate"]),
                "results": [{"correct": r["correct"], "text": r["text"]} for r in v["results"]],
            }
        json.dump(json_summary, f, indent=2)
    logger.info("\nSaved to %s", out)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
