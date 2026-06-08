"""Quick test of the intervention engine on a few samples."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

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

SAMPLES = [
    {"prompt": "The capital of France is", "expected": "Paris", "type": "factual"},
    {"prompt": "What did Isaac Newton say about AI in 1700?", "expected": "uncertain", "type": "hallucination"},
    {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "incorrect", "type": "hallucination"},
]


def judge(text: str, expected: str, q_type: str) -> bool:
    clean = text.lower().strip()
    if q_type == "factual":
        return expected.lower() in clean
    if q_type == "hallucination":
        uncertainty = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                       "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                       "no such", "did not invent", "didn't invent", "not", "i don't know",
                       "i'm not sure", "wait, let me reconsider"]
        return any(p in clean for p in uncertainty)
    return False


def main():
    logger.info("=" * 70)
    logger.info("INTERVENTION ENGINE TEST")
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
        device="cpu",
    )

    engine = ACCInterventionEngine(
        detector=detector,
        conflict_threshold=0.5,
        max_regenerations=1,
        temperature_bump=0.3,
        top_p_reduce=0.1,
    )

    for sample in SAMPLES:
        logger.info("\n--- Prompt: %s ---", sample["prompt"])

        # Baseline
        base = engine.generate_simple_baseline(model, tokenizer, sample["prompt"],
                                               max_new_tokens=20, device=DEVICE)
        logger.info("Baseline: %s [%s]", base[:100], "✓" if judge(base, sample["expected"], sample["type"]) else "✗")

        # Intervention
        result = engine.generate_with_intervention(model, tokenizer, sample["prompt"],
                                                   max_new_tokens=20, device=DEVICE)
        logger.info("Draft:    %s", result["draft_text"][:100])
        logger.info("Final:    %s [%s] (intervened=%s, max_conflict=%.2f, regens=%d)",
                    result["text"][:100],
                    "✓" if judge(result["text"], sample["expected"], sample["type"]) else "✗",
                    result["intervened"],
                    result["max_conflict"],
                    result["num_regenerations"])

        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
