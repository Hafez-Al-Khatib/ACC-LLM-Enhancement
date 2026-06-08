"""Evaluate the HaluEval-trained detector on PubMedQA + known questions.

Compares:
  - Baseline (no ACC)
  - Entropy-only (calibrated threshold)
  - HaluEval detector (logits-biasing intervention, no text markers)
"""

from __future__ import annotations

import json
import logging
import os
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
from src.halueval_detector import HaluEvalDetector
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig
from src.acc_layer import EntropyMonitor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 20
NUM_PUBMED_SAMPLES = 8
JUDGE_MAX_TOKENS = 5

KNOWN_QUESTIONS = [
    {"question": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"question": "Water boils at a temperature of", "expected": "100", "type": "factual"},
    {"question": "What did Isaac Newton say about artificial intelligence in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"question": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
]


def load_pubmedqa(n_samples: int = 8) -> List[Dict]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    indices = np.random.RandomState(42).choice(len(ds), min(n_samples, len(ds)), replace=False)
    samples = []
    for i in indices:
        item = ds[int(i)]
        question = item.get("question", "")
        context = item.get("context", "")
        if isinstance(context, list):
            context = " ".join(context)
        final_decision = item.get("final_decision", "")
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        samples.append({"prompt": prompt, "question": question, "expected": final_decision, "type": "pubmedqa"})
    return samples


def extract_yes_no_maybe(text: str) -> str:
    text_lower = text.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text_lower.startswith(label):
            return label
    words = text_lower.split()[:10]
    for label in ["yes", "no", "maybe"]:
        if label in words:
            return label
    return "other"


def judge_correct(generated: str, expected: str, q_type: str) -> bool:
    clean = generated.lower().strip()
    expected_clean = expected.lower().strip()
    if q_type == "pubmedqa":
        return extract_yes_no_maybe(clean) == expected_clean
    if q_type == "factual":
        return expected_clean in clean
    if q_type == "hallucination":
        uncertainty = ["did not", "didn't", "never", "not possible", "impossible",
                       "incorrect", "false", "no evidence", "not true", "myth",
                       "there is no", "there was no", "i don't know", "uncertain",
                       "cannot", "can't", "could not", "no such", "did not invent",
                       "didn't invent"]
        return any(p in clean for p in uncertainty)
    return False


def run_baseline(model, tokenizer, prompt: str, device: str) -> Tuple[str, Dict]:
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

    return text, {"mean_entropy": np.mean(entropies), "max_entropy": max(entropies)}


def run_with_halueval_detector(
    model, tokenizer, prompt: str, detector: HaluEvalDetector, device: str
) -> Tuple[str, Dict]:
    """Generate with HaluEval detector using logits biasing (no text markers)."""
    layer_indices = [-1, -4, -8, -12, -16, -20, -24, -28]

    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="warning",  # Don't insert markers; just log decisions
        threshold=3.9,
        mode="absolute",
        use_conflict_detector=True,
        use_realtime_conflict_detector=True,
        conflict_detector=detector,
        conflict_layer_indices=layer_indices,
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=3.9,
            conflict_score_threshold=0.7,
            dual_signal_regenerate=False,
            marker_config=MarkerConfig(hallucination="", contradiction="", uncertain=""),
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
        "n_warnings": sum(1 for d in decisions if d["action"] == "warning"),
        "mean_entropy": np.mean([d.get("entropy", 0.0) for d in decisions]),
        "max_entropy": max((d.get("entropy") or 0.0 for d in decisions), default=0.0),
        "max_conflict": max((d.get("conflict_score") or 0.0 for d in decisions), default=0.0),
        "mean_conflict": np.mean([d.get("conflict_score") or 0.0 for d in decisions]),
    }
    return text, stats


def evaluate():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("HALUEVAL DETECTOR EVALUATION")
    logger.info("Device: %s", device.upper())
    logger.info("=" * 70)

    logger.info("\nLoading model...")
    model_name = "models/qwen2.5-1.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True
    )
    if device == "xpu":
        model = model.to("xpu")
    logger.info("Model loaded on %s", next(model.parameters()).device)

    # Load HaluEval detector
    layer_pairs = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]
    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=layer_pairs,
        checkpoint_path="adapters/halueval_detector.pt",
        device=device,
    )
    logger.info("Loaded HaluEval detector: %d params", sum(p.numel() for p in detector.parameters()))

    # Build dataset
    pubmed_samples = load_pubmedqa(NUM_PUBMED_SAMPLES)
    known_samples = [{"prompt": q["question"], "question": q["question"], "expected": q["expected"], "type": q["type"]} for q in KNOWN_QUESTIONS]
    all_samples = pubmed_samples + known_samples

    logger.info("Evaluating %d samples (%d PubMedQA + %d known)...", len(all_samples), len(pubmed_samples), len(known_samples))

    baseline_results = []
    detector_results = []

    for sample in tqdm(all_samples, desc="Evaluating"):
        # Baseline
        base_text, base_stats = run_baseline(model, tokenizer, sample["prompt"], device)
        base_correct = judge_correct(base_text, sample["expected"], sample["type"])
        baseline_results.append({"correct": base_correct, "text": base_text, "type": sample["type"], **base_stats})

        # HaluEval detector
        det_text, det_stats = run_with_halueval_detector(model, tokenizer, sample["prompt"], detector, device)
        det_correct = judge_correct(det_text, sample["expected"], sample["type"])
        detector_results.append({"correct": det_correct, "text": det_text, "type": sample["type"], **det_stats})

        if device == "xpu":
            torch.xpu.empty_cache()

    # Aggregate
    def summarize(name, results):
        acc = np.mean([r["correct"] for r in results])
        mean_h = np.mean([r["mean_entropy"] for r in results])
        max_h = np.mean([r["max_entropy"] for r in results])
        flags = np.mean([r.get("n_flags", 0) + r.get("n_warnings", 0) for r in results])
        mean_cs = np.mean([r.get("mean_conflict", 0.0) for r in results])
        logger.info("  %-20s | Acc: %.1f%% | MeanH: %.3f | MaxH: %.3f | Flags: %.1f | AvgCS: %.3f",
                    name, acc * 100, mean_h, max_h, flags, mean_cs)
        return {"accuracy": acc, "mean_entropy": mean_h, "max_entropy": max_h, "avg_flags": flags, "avg_conflict": mean_cs}

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info("  %-20s | Acc    | MeanH  | MaxH   | Flags  | AvgCS", "Config")
    logger.info("  " + "-" * 70)
    baseline_met = summarize("Baseline", baseline_results)
    detector_met = summarize("HaluEval Detector", detector_results)

    logger.info("\nImprovement: %+.1f%% accuracy", (detector_met["accuracy"] - baseline_met["accuracy"]) * 100)

    # Per-type breakdown
    logger.info("\nPer-type breakdown:")
    for q_type in ["pubmedqa", "factual", "hallucination"]:
        type_base = [r for r in baseline_results if r["type"] == q_type]
        type_det = [r for r in detector_results if r["type"] == q_type]
        if type_base:
            logger.info("\n  %s (%d samples):", q_type)
            summarize("  Baseline", type_base)
            summarize("  Detector", type_det)

    # Save
    out_path = Path("results/halueval_detector_evaluation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "samples": [{"question": s["question"], "expected": s["expected"], "type": s["type"]} for s in all_samples],
            "baseline": baseline_results,
            "detector": detector_results,
            "metrics": {"baseline": baseline_met, "detector": detector_met},
        }, f, indent=2)
    logger.info("\nResults saved to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    evaluate()
