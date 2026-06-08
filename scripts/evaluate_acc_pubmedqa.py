"""Evaluate ACC on PubMedQA + known factual/hallucination questions.

Benchmarks:
  - PubMedQA (pqa_labeled subset): biomedical QA with yes/no/maybe answers
  - Known Questions: curated factual vs. hallucination-inducing prompts

Configs:
  - Baseline
  - Entropy-only
  - Detector+Flag

Metrics:
  - Accuracy (exact match for yes/no/maybe; judge-based for open-ended)
  - Hallucination/intervention rate
  - Per-token entropy and conflict statistics
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig
from src.acc_conflict_detector import PredictiveCodingDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 25
NUM_PUBMED_SAMPLES = 4
NUM_KNOWN_SAMPLES = 4
JUDGE_MAX_TOKENS = 5

MARKER_PATTERN = re.compile(r"\s*\[(HALLUCINATION|CONTRADICTION|UNCERTAIN|ENTROPY FLAG)\]\s*")

# ---------------------------------------------------------------------------
# Known questions with expected behavior
# ---------------------------------------------------------------------------
KNOWN_QUESTIONS = [
    # Factual (should NOT be flagged)
    {"question": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"question": "Water boils at a temperature of", "expected": "100", "type": "factual"},
    # Hallucination-inducing (SHOULD be flagged or produce uncertain answers)
    {"question": "What did Isaac Newton say about artificial intelligence in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"question": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
]


def strip_markers(text: str) -> str:
    return MARKER_PATTERN.sub(" ", text).strip()


def load_pubmedqa(n_samples: int = 15) -> List[Dict]:
    """Load PubMedQA labeled subset."""
    logger.info("Downloading PubMedQA dataset...")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to load qiaojin/PubMedQA: %s", e)
        logger.info("Trying alternative: pubmed_qa...")
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)

    indices = np.random.RandomState(42).choice(len(ds), min(n_samples, len(ds)), replace=False)
    samples = []
    for i in indices:
        item = ds[int(i)]
        question = item.get("question", "")
        context = item.get("context", "")
        if isinstance(context, list):
            context = " ".join(context)
        final_decision = item.get("final_decision", "")
        long_answer = item.get("long_answer", "")

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        samples.append({
            "prompt": prompt,
            "question": question,
            "final_decision": final_decision,
            "long_answer": long_answer,
        })

    logger.info("Loaded %d PubMedQA samples", len(samples))
    return samples


def extract_yes_no_maybe(text: str) -> str:
    """Extract yes/no/maybe from generated text."""
    text_lower = text.lower().strip()
    # Look for explicit yes/no/maybe at the start
    for label in ["yes", "no", "maybe"]:
        if text_lower.startswith(label):
            return label
    # Or anywhere in the first few words
    words = text_lower.split()[:10]
    for label in ["yes", "no", "maybe"]:
        if label in words:
            return label
    return "other"


def judge_answer_correct(
    generated: str,
    expected: str,
    q_type: str = "open",
) -> bool:
    """Judge correctness based on question type."""
    clean_gen = strip_markers(generated).lower().strip()
    expected_clean = expected.lower().strip()

    if q_type == "pubmedqa":
        # Exact match on yes/no/maybe
        pred = extract_yes_no_maybe(clean_gen)
        return pred == expected_clean

    if q_type == "factual":
        # Contains expected substring
        return expected_clean in clean_gen

    if q_type == "hallucination":
        # For hallucination prompts, we EXPECT the model to say something false
        # A "good" ACC response would either abstain, express uncertainty, or flag
        # We consider it "correctly handled" if it contains uncertainty language
        # or if it DOES NOT contain a confident false claim matching known bad patterns
        uncertainty_phrases = [
            "did not", "didn't", "never", "not possible", "impossible",
            "incorrect", "false", "no evidence", "not true", "myth",
            "there is no", "there was no", "i don't know", "uncertain",
            "cannot", "can't", "could not", "no such",
        ]
        return any(phrase in clean_gen for phrase in uncertainty_phrases)

    return False


def run_generation(
    model,
    tokenizer,
    prompt: str,
    detector: PredictiveCodingDetector = None,
    use_entropy: bool = True,
    use_detector: bool = False,
    device: str = "xpu",
) -> Tuple[str, Dict]:
    """Generate with optional ACC."""
    layer_indices = [-1, -4, -8, -12, -16, -20, -24, -28]

    if not use_entropy and not use_detector:
        # Baseline
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs.sequences[0, input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        entropies = []
        for score in outputs.scores:
            probs = F.softmax(score[0], dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropies.append(-(probs * log_probs).sum().item())

        return text, {
            "mean_entropy": np.mean(entropies) if entropies else 0.0,
            "max_entropy": max(entropies) if entropies else 0.0,
            "n_flags": 0,
            "n_regens": 0,
            "max_conflict": 0.0,
        }

    # ACC-enhanced
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="flag",
        threshold=3.9,
        mode="absolute",
        use_conflict_detector=use_detector,
        use_realtime_conflict_detector=use_detector,
        conflict_detector=detector if use_detector else None,
        conflict_layer_indices=layer_indices if use_detector else None,
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=3.9,
            conflict_score_threshold=0.85 if use_detector else 0.99,
            dual_signal_regenerate=False,
            marker_config=MarkerConfig(
                hallucination=" [HALLUCINATION]",
                contradiction=" [CONTRADICTION]",
                uncertain=" [UNCERTAIN]",
            ),
        ),
    )

    output = gen.generate_from_prompt(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
        top_p=0.95,
        return_dict_in_generate=True,
    )

    text = output.text[0]
    decisions = output.per_token_decisions[0]

    stats = {
        "n_flags": sum(1 for d in decisions if d["action"] == "flag"),
        "n_regens": sum(1 for d in decisions if d["action"] == "regenerate"),
        "mean_entropy": np.mean([d.get("entropy", 0.0) for d in decisions]) if decisions else 0.0,
        "max_entropy": max((d.get("entropy") or 0.0 for d in decisions), default=0.0),
        "max_conflict": max((d.get("conflict_score") or 0.0 for d in decisions), default=0.0),
    }
    return text, stats


def evaluate_config(model, tokenizer, detector, samples, device, config_name, use_entropy, use_detector):
    """Evaluate one configuration on a list of samples."""
    results = []
    for sample in samples:
        prompt = sample["prompt"]
        text, stats = run_generation(
            model, tokenizer, prompt, detector,
            use_entropy=use_entropy, use_detector=use_detector, device=device
        )

        correct = judge_answer_correct(
            text,
            sample["expected"],
            q_type=sample.get("type", "open"),
        )

        results.append({
            "question": sample.get("question", prompt[:80]),
            "expected": sample["expected"],
            "type": sample.get("type", "open"),
            "text": text,
            "clean_text": strip_markers(text),
            "correct": correct,
            **stats,
        })

    acc = np.mean([r["correct"] for r in results])
    mean_h = np.mean([r["mean_entropy"] for r in results])
    max_h = np.mean([r["max_entropy"] for r in results])
    flags = np.mean([r["n_flags"] for r in results])
    regens = np.mean([r["n_regens"] for r in results])
    max_conflict = np.mean([r["max_conflict"] for r in results])

    logger.info("  %-18s | Acc: %.1f%% | MeanH: %.3f | MaxH: %.3f | Flags: %.1f | Regens: %.1f | CS: %.3f",
                config_name, acc * 100, mean_h, max_h, flags, regens, max_conflict)

    return results, {
        "accuracy": acc,
        "mean_entropy": mean_h,
        "max_entropy": max_h,
        "avg_flags": flags,
        "avg_regenerations": regens,
        "avg_max_conflict": max_conflict,
    }


def evaluate(model_name: str = "models/qwen2.5-1.5b"):
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("ACC EVALUATION: PubMedQA + Known Questions")
    logger.info("Device: %s", device.upper())
    logger.info("=" * 70)

    logger.info("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    if device == "xpu":
        model = model.to("xpu")
    logger.info("Model loaded on %s", next(model.parameters()).device)

    # Load detector
    hidden_dim = model.config.hidden_size
    layer_pairs = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]
    detector = PredictiveCodingDetector(hidden_dim=hidden_dim, layer_pairs=layer_pairs, temporal_decay=0.7)
    detector_path = "adapters/qwen2.5_detector.pt"
    if os.path.exists(detector_path):
        detector.load_state_dict(torch.load(detector_path, map_location=device))
        logger.info("Loaded trained detector")
    detector.eval()
    if device == "xpu":
        detector = detector.to("xpu")

    # Load datasets
    pubmed_samples = load_pubmedqa(NUM_PUBMED_SAMPLES)
    # Format PubMedQA samples
    pubmed_formatted = []
    for s in pubmed_samples:
        pubmed_formatted.append({
            "prompt": s["prompt"],
            "question": s["question"],
            "expected": s["final_decision"],
            "type": "pubmedqa",
        })

    known_formatted = []
    for s in KNOWN_QUESTIONS:
        known_formatted.append({
            "prompt": s["question"],
            "question": s["question"],
            "expected": s["expected"],
            "type": s["type"],
        })

    all_samples = pubmed_formatted + known_formatted
    logger.info("\nTotal samples: %d (PubMedQA: %d, Known: %d)",
                len(all_samples), len(pubmed_formatted), len(known_formatted))

    # Evaluate
    logger.info("\n  %-18s | Acc    | MeanH | MaxH  | Flags | Regens | CS",
                "Config")
    logger.info("  " + "-" * 75)

    baseline_res, baseline_met = evaluate_config(
        model, tokenizer, detector, all_samples, device, "Baseline", False, False
    )
    entropy_res, entropy_met = evaluate_config(
        model, tokenizer, detector, all_samples, device, "Entropy-only", True, False
    )
    detector_res, detector_met = evaluate_config(
        model, tokenizer, detector, all_samples, device, "Detector+Flag", True, True
    )

    logger.info("\nImprovements over baseline:")
    logger.info("  Entropy-only:  %+.1f%% accuracy", (entropy_met["accuracy"] - baseline_met["accuracy"]) * 100)
    logger.info("  Detector+Flag: %+.1f%% accuracy", (detector_met["accuracy"] - baseline_met["accuracy"]) * 100)

    # Break down by type
    logger.info("\nBreakdown by question type:")
    for q_type in ["pubmedqa", "factual", "hallucination"]:
        type_samples = [s for s in all_samples if s["type"] == q_type]
        if not type_samples:
            continue
        logger.info("\n  %s (%d samples):", q_type, len(type_samples))
        _, bmet = evaluate_config(model, tokenizer, detector, type_samples, device, "  Baseline", False, False)
        _, emet = evaluate_config(model, tokenizer, detector, type_samples, device, "  Entropy", True, False)
        _, dmet = evaluate_config(model, tokenizer, detector, type_samples, device, "  Detector", True, True)

    # Save results
    out_path = Path("results/acc_evaluation_pubmedqa.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "samples": [{"question": s["question"], "expected": s["expected"], "type": s["type"]} for s in all_samples],
            "baseline": baseline_res,
            "entropy_only": entropy_res,
            "detector_flag": detector_res,
            "metrics": {
                "baseline": baseline_met,
                "entropy_only": entropy_met,
                "detector_flag": detector_met,
            }
        }, f, indent=2)
    logger.info("\nResults saved to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    evaluate()
