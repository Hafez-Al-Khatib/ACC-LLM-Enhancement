"""Micro evaluation: 5 samples, baseline + entropy + ACC-detector (simple mode).

ACC-detector runs WITHOUT hooks (post-hoc on hidden states) to avoid memory issues.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.halueval_detector import HaluEvalDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
MAX_NEW_TOKENS = 20

SAMPLES = [
    {"prompt": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"prompt": "Water boils at a temperature of", "expected": "100", "type": "factual"},
    {"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
    {"prompt": "The first President of the United States was", "expected": "Washington", "type": "factual"},
]


def judge(text: str, expected: str, q_type: str) -> bool:
    clean = text.lower().strip()
    exp = expected.lower().strip()
    if q_type == "factual":
        return exp in clean
    if q_type == "hallucination":
        uncertainty = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                       "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                       "no such", "did not invent", "didn't invent", "not"]
        return any(p in clean for p in uncertainty)
    return False


def entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return (-(probs * log_probs).sum()).item()


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
    entropies = [entropy(s[0]) for s in out.scores]
    flags = sum(1 for h in entropies if h > threshold)
    return text, flags, max(entropies) if entropies else 0.0


def run_acc_simple(model, tokenizer, prompt: str, detector: HaluEvalDetector):
    """Run ACC detector in post-hoc mode: get hidden states, then detect."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    # Collect hidden states using forward pass (no hooks)
    hidden_states_list = []
    current_input_ids = inputs["input_ids"]

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(current_input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
            last_token_hs = {}
            for layer_idx, hs in enumerate(hidden_states):
                last_token_hs[layer_idx - len(hidden_states)] = hs[0, -1, :].detach().cpu()
                last_token_hs[layer_idx] = hs[0, -1, :].detach().cpu()

            # Run detector
            detector_out = detector.forward(last_token_hs)
            conflict_score = float(detector_out[2][0].item())
            hidden_states_list.append({"conflict": conflict_score})

            # Sample next token
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(current_input_ids[0, input_len:], skip_special_tokens=True)
    flags = sum(1 for h in hidden_states_list if h["conflict"] > 0.5)
    max_conflict = max((h["conflict"] for h in hidden_states_list), default=0.0)
    return text, flags, max_conflict


def main():
    logger.info("=" * 60)
    logger.info("MICRO EVALUATION (5 samples)")
    logger.info("=" * 60)

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
        device="cpu",  # Run detector on CPU to save XPU memory
    )

    baseline_results = []
    entropy_results = []
    acc_results = []

    for i, sample in enumerate(SAMPLES, 1):
        logger.info("\n--- Sample %d: %s ---", i, sample["type"])
        prompt = sample["prompt"]

        # Baseline
        base_text = run_baseline(model, tokenizer, prompt)
        base_correct = judge(base_text, sample["expected"], sample["type"])
        baseline_results.append({"correct": base_correct, "text": base_text})
        logger.info("Baseline: %s [%s]", base_text[:80], "✓" if base_correct else "✗")

        # Entropy
        ent_text, ent_flags, ent_max = run_entropy(model, tokenizer, prompt)
        ent_correct = judge(ent_text, sample["expected"], sample["type"])
        entropy_results.append({"correct": ent_correct, "text": ent_text, "flags": ent_flags, "max_entropy": ent_max})
        logger.info("Entropy:  %s [%s] (flags=%d, max_H=%.2f)", ent_text[:80], "✓" if ent_correct else "✗", ent_flags, ent_max)

        # ACC
        acc_text, acc_flags, acc_max = run_acc_simple(model, tokenizer, prompt, detector)
        acc_correct = judge(acc_text, sample["expected"], sample["type"])
        acc_results.append({"correct": acc_correct, "text": acc_text, "flags": acc_flags, "max_conflict": acc_max})
        logger.info("ACC:      %s [%s] (flags=%d, max_C=%.2f)", acc_text[:80], "✓" if acc_correct else "✗", acc_flags, acc_max)

        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    b_acc = sum(r["correct"] for r in baseline_results) / len(baseline_results)
    e_acc = sum(r["correct"] for r in entropy_results) / len(entropy_results)
    a_acc = sum(r["correct"] for r in acc_results) / len(acc_results)
    logger.info("Baseline:     %.0f%% (%d/5)", b_acc * 100, int(b_acc * 5))
    logger.info("Entropy:      %.0f%% (%d/5)", e_acc * 100, int(e_acc * 5))
    logger.info("ACC-Detector: %.0f%% (%d/5)", a_acc * 100, int(a_acc * 5))
    logger.info("Improvement:  %+.0f%% over baseline", (a_acc - b_acc) * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
