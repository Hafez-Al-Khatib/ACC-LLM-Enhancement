"""Minimal but rigorous evaluation: baseline, entropy, ACC-detector.
Benchmarks: PubMedQA + known questions.
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
MAX_NEW_TOKENS = 20


def entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return (-(probs * log_probs).sum()).item()


def load_all_samples() -> List[Dict]:
    samples = []

    # PubMedQA
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        indices = np.random.RandomState(42).choice(len(ds), min(20, len(ds)), replace=False)
        for i in indices:
            item = ds[int(i)]
            ctx = item.get("context", "")
            if isinstance(ctx, list):
                ctx = " ".join(ctx)
            samples.append({
                "prompt": f"Context: {ctx}\nQuestion: {item['question']}\nAnswer:",
                "expected": item.get("final_decision", ""),
                "type": "pubmedqa",
            })
    except Exception as e:
        logger.warning("PubMedQA: %s", e)

    # Known factual
    samples.append({"prompt": "The capital of France is", "expected": "Paris", "type": "factual"})
    samples.append({"prompt": "Water boils at a temperature of", "expected": "100", "type": "factual"})
    samples.append({"prompt": "The square root of 144 is", "expected": "12", "type": "factual"})

    # Known hallucination
    samples.append({"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"})
    samples.append({"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"})
    samples.append({"prompt": "Describe the lunar landing in 1845", "expected": "incorrect", "type": "hallucination"})

    return samples


def judge(text: str, expected: str, q_type: str) -> bool:
    clean = text.lower().strip()
    exp = expected.lower().strip()
    if q_type == "pubmedqa":
        pred = "yes" if clean.startswith("yes") or "yes" in clean.split()[:5] else \
               "no" if clean.startswith("no") or "no" in clean.split()[:5] else "maybe"
        return pred == exp
    if q_type == "factual":
        return exp in clean
    if q_type == "hallucination":
        uncertainty = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                       "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                       "no such", "did not invent", "didn't invent"]
        return any(p in clean for p in uncertainty)
    return False


def run_baseline(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


def run_entropy(model, tokenizer, prompt: str, threshold: float = 3.9) -> Tuple[str, int, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=0.8, top_p=0.95, return_dict_in_generate=True,
                             output_scores=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out.sequences[0, input_len:], skip_special_tokens=True)
    entropies = [entropy(s[0]) for s in out.scores]
    flags = sum(1 for h in entropies if h > threshold)
    return text, flags, max(entropies) if entropies else 0.0


def run_acc(model, tokenizer, prompt: str, detector: HaluEvalDetector) -> Tuple[str, int, float]:
    gen = ACCEnhancedGenerator(
        model=model, tokenizer=tokenizer, action="warning", threshold=3.9, mode="absolute",
        use_conflict_detector=True, use_realtime_conflict_detector=True,
        conflict_detector=detector, conflict_layer_indices=[-1, -4, -8, -12, -16, -20, -24, -28],
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=3.9, conflict_score_threshold=0.7, dual_signal_regenerate=False,
            marker_config=MarkerConfig(hallucination="", contradiction="", uncertain=""),
        ),
    )
    output = gen.generate_from_prompt(prompt, max_new_tokens=MAX_NEW_TOKENS,
                                      temperature=0.8, top_p=0.95, return_dict_in_generate=True)
    text = output.text[0]
    decisions = output.per_token_decisions[0]
    flags = sum(1 for d in decisions if d.get("action") in ["flag", "warning"])
    max_conflict = max((d.get("conflict_score") or 0.0 for d in decisions), default=0.0)
    return text, flags, max_conflict


def main():
    logger.info("=" * 70)
    logger.info("MINIMAL EVALUATION FRAMEWORK")
    logger.info("=" * 70)

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
        device=DEVICE,
    )

    samples = load_all_samples()
    logger.info("Evaluating %d samples\n", len(samples))

    baseline_results = []
    entropy_results = []
    acc_results = []

    for sample in tqdm(samples, desc="Evaluating"):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]

        # Baseline
        base_text = run_baseline(model, tokenizer, prompt)
        baseline_results.append({
            "correct": judge(base_text, expected, q_type),
            "text": base_text,
            "type": q_type,
        })

        # Entropy
        ent_text, ent_flags, ent_max = run_entropy(model, tokenizer, prompt)
        entropy_results.append({
            "correct": judge(ent_text, expected, q_type),
            "text": ent_text,
            "flags": ent_flags,
            "max_entropy": ent_max,
            "type": q_type,
        })

        # ACC
        acc_text, acc_flags, acc_max = run_acc(model, tokenizer, prompt, detector)
        acc_results.append({
            "correct": judge(acc_text, expected, q_type),
            "text": acc_text,
            "flags": acc_flags,
            "max_conflict": acc_max,
            "type": q_type,
        })

        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    def summarize(name, results):
        acc = np.mean([r["correct"] for r in results])
        flags = np.mean([r.get("flags", 0) for r in results])
        logger.info("  %-15s | Accuracy: %.1f%% | Avg Flags: %.1f", name, acc * 100, flags)
        return {"accuracy": acc, "avg_flags": flags}

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    baseline_met = summarize("Baseline", baseline_results)
    entropy_met = summarize("Entropy", entropy_results)
    acc_met = summarize("ACC-Detector", acc_results)

    logger.info("\nImprovements:")
    logger.info("  Entropy vs Baseline:    %+.1f%%", (entropy_met["accuracy"] - baseline_met["accuracy"]) * 100)
    logger.info("  ACC vs Baseline:        %+.1f%%", (acc_met["accuracy"] - baseline_met["accuracy"]) * 100)
    logger.info("  ACC vs Entropy:         %+.1f%%", (acc_met["accuracy"] - entropy_met["accuracy"]) * 100)

    # Per-type
    logger.info("\nPer-type breakdown:")
    for q_type in ["pubmedqa", "factual", "hallucination"]:
        b = [r for r in baseline_results if r["type"] == q_type]
        e = [r for r in entropy_results if r["type"] == q_type]
        a = [r for r in acc_results if r["type"] == q_type]
        if b:
            logger.info("\n  %s (%d samples):", q_type)
            summarize("    Baseline", b)
            summarize("    Entropy", e)
            summarize("    ACC", a)

    out = Path("results/evaluation_minimal.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "samples": [{"prompt": s["prompt"], "expected": s["expected"], "type": s["type"]} for s in samples],
            "baseline": baseline_results,
            "entropy": entropy_results,
            "acc": acc_results,
        }, f, indent=2)
    logger.info("\nSaved to: %s", out)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
