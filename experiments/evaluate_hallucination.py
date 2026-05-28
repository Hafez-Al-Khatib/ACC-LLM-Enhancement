"""Evaluate generated text for hallucinations against ground-truth answers.

Metrics:
  - Token-level hallucination F1 (via entailment heuristics)
  - Contradiction rate
  - Calibration error (confidence vs. accuracy)
  - Perplexity on ground-truth answer

Input: JSONL with records containing {prompt, ground_truth_output, generated_text,
                                       token_probs (optional), per_token_entropy (optional)}
Output: JSON report with aggregate and per-sample metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def lexical_overlap_score(generated: str, ground_truth: str) -> float:
    """Simple token-overlap score between 0 and 1."""
    gen_tokens = set(normalize(generated).split())
    gt_tokens = set(normalize(ground_truth).split())
    if not gen_tokens or not gt_tokens:
        return 0.0
    inter = len(gen_tokens & gt_tokens)
    return inter / max(len(gen_tokens), len(gt_tokens))


def detect_contradiction_heuristic(generated: str, ground_truth: str) -> bool:
    """Crude heuristic: look for explicit negation of key phrases."""
    gen_norm = normalize(generated)
    gt_norm = normalize(ground_truth)

    # Simple negation patterns
    negation_words = {"not", "no", "never", "none", "cannot", "impossible"}
    gen_has_neg = any(w in gen_norm for w in negation_words)
    gt_has_neg = any(w in gt_norm for w in negation_words)

    # If ground truth is positive and generation has negation, flag
    if not gt_has_neg and gen_has_neg:
        # Check if they share content words
        content_words = [w for w in gt_norm.split() if len(w) > 3]
        shared = sum(1 for w in content_words if w in gen_norm)
        if shared >= max(1, len(content_words) // 2):
            return True
    return False


def token_level_f1(generated: str, ground_truth: str) -> Dict[str, float]:
    """Treat hallucinated tokens as those in generated but not in ground truth."""
    gen_tokens = set(normalize(generated).split())
    gt_tokens = set(normalize(ground_truth).split())

    tp = len(gen_tokens & gt_tokens)
    fp = len(gen_tokens - gt_tokens)
    fn = len(gt_tokens - gen_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calibration_error(token_probs: Optional[List[float]], is_correct: bool) -> Optional[float]:
    """Absolute difference between mean confidence and binary correctness."""
    if not token_probs:
        return None
    mean_conf = sum(token_probs) / len(token_probs)
    return abs(mean_conf - float(is_correct))


def perplexity(token_probs: Optional[List[float]]) -> Optional[float]:
    """Compute perplexity from token probabilities."""
    if not token_probs or any(p <= 0 for p in token_probs):
        return None
    log_probs = [math.log(p) for p in token_probs]
    return math.exp(-sum(log_probs) / len(log_probs))


def evaluate_record(record: Dict) -> Dict:
    gen = record.get("generated_text", "")
    gt = record.get("ground_truth_output", "")
    token_probs = record.get("token_probs")

    overlap = lexical_overlap_score(gen, gt)
    contradict = detect_contradiction_heuristic(gen, gt)
    f1_scores = token_level_f1(gen, gt)
    cal_err = calibration_error(token_probs, overlap > 0.5)
    ppl = perplexity(token_probs)

    # Hallucination flag: low overlap and no contradiction detected = likely hallucinated
    is_hallucinated = overlap < 0.3 and not contradict

    return {
        "prompt": record.get("prompt", ""),
        "generated_text": gen,
        "ground_truth_output": gt,
        "lexical_overlap": overlap,
        "contradiction_detected": contradict,
        "is_hallucinated": is_hallucinated,
        "token_level_precision": f1_scores["precision"],
        "token_level_recall": f1_scores["recall"],
        "token_level_f1": f1_scores["f1"],
        "calibration_error": cal_err,
        "perplexity": ppl,
    }


def aggregate(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {}

    hallucinated = sum(1 for r in results if r["is_hallucinated"])
    contradictions = sum(1 for r in results if r["contradiction_detected"])
    cal_errors = [r["calibration_error"] for r in results if r["calibration_error"] is not None]
    ppls = [r["perplexity"] for r in results if r["perplexity"] is not None]

    return {
        "n_samples": n,
        "hallucination_rate": hallucinated / n,
        "contradiction_rate": contradictions / n,
        "mean_lexical_overlap": sum(r["lexical_overlap"] for r in results) / n,
        "mean_token_precision": sum(r["token_level_precision"] for r in results) / n,
        "mean_token_recall": sum(r["token_level_recall"] for r in results) / n,
        "mean_token_f1": sum(r["token_level_f1"] for r in results) / n,
        "mean_calibration_error": sum(cal_errors) / len(cal_errors) if cal_errors else None,
        "mean_perplexity": sum(ppls) / len(ppls) if ppls else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination metrics")
    parser.add_argument("--input", required=True, help="Input JSONL with generated results")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    evaluated = [evaluate_record(r) for r in records]
    agg = aggregate(evaluated)

    report = {
        "aggregate": agg,
        "per_sample": evaluated,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote evaluation report to {out_path}")
    print(f"  Samples: {agg['n_samples']}")
    print(f"  Hallucination rate: {agg['hallucination_rate']:.3f}")
    print(f"  Contradiction rate: {agg['contradiction_rate']:.3f}")
    print(f"  Mean token F1: {agg['mean_token_f1']:.3f}")
    print(f"  Mean calibration error: {agg['mean_calibration_error']:.3f}" if agg["mean_calibration_error"] is not None else "  Mean calibration error: N/A")


if __name__ == "__main__":
    main()
