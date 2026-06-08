"""Smoke test: Load Mistral 7B on Intel Arc XPU and run ACC-enhanced generation.

Usage:
    .venv\Scripts\activate
    python scripts/smoke_test_mistral_xpu.py

This validates:
  - Model loads on XPU without OOM
  - ACCEnhancedGenerator works with real model hidden states
  - Conflict detector produces labels and scores
  - Per-token decisions are populated
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# Allow importing src/ from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.model_utils import load_model, load_tokenizer, make_bnb_config
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig
from src.acc_conflict_detector import PredictiveCodingDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    model_path = "models/mistral_7b"
    device = "xpu"

    logger.info("=" * 60)
    logger.info("Mistral 7B + ACC Smoke Test on Intel Arc XPU")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path, local_files_only=True)

    # ------------------------------------------------------------------
    # 2. Load model (4-bit QLoRA to fit in 16GB Arc GPU)
    # ------------------------------------------------------------------
    logger.info("Loading Mistral 7B in 4-bit NF4 on %s...", device)
    t0 = time.time()
    bnb = make_bnb_config({
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    })
    model = load_model(
        model_path,
        bnb_config=bnb,
        device=device,
        local_files_only=True,
    )
    load_time = time.time() - t0
    logger.info("Model loaded in %.1f seconds on %s", load_time, next(model.parameters()).device)

    # ------------------------------------------------------------------
    # 3. Build conflict detector
    # ------------------------------------------------------------------
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    logger.info("Model config: hidden_dim=%d, num_layers=%d", hidden_dim, num_layers)

    detector = PredictiveCodingDetector(
        hidden_dim=hidden_dim,
        layer_pairs=[(-12, -8), (-8, -4), (-4, -1)],
        temporal_decay=0.7,
    )
    detector.eval()
    detector = detector.to(device)

    # ------------------------------------------------------------------
    # 4. Build ACC generator
    # ------------------------------------------------------------------
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="flag",
        threshold=1.5,
        mode="absolute",
        use_conflict_detector=True,
        use_realtime_conflict_detector=True,
        conflict_detector=detector,
        conflict_layer_indices=[-1, -4, -8, -12],
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=1.5,
            conflict_score_threshold=0.7,
            dual_signal_regenerate=True,
        ),
    )

    # ------------------------------------------------------------------
    # 5. Generate with monitoring
    # ------------------------------------------------------------------
    torch.xpu.empty_cache()  # Free fragmented memory before generation
    prompt = "The capital of France is"
    logger.info("Generating from prompt: '%s'", prompt)

    t0 = time.time()
    result = gen.generate_from_prompt(
        prompt,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        return_dict_in_generate=True,
    )
    gen_time = time.time() - t0

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Generation completed in %.2f seconds", gen_time)
    logger.info("Output text: %s", result.text[0])
    logger.info("Tokens generated: %d", len(result.per_token_entropy[0]))
    logger.info("Uncertain steps: %s", result.uncertain_steps[0])
    logger.info("Regenerations: %d", result.regenerations[0])
    logger.info("Confidence score: %.3f", result.confidence_score[0])

    # Decision trace
    decisions = result.per_token_decisions[0]
    logger.info("Per-token decisions:")
    for i, d in enumerate(decisions):
        cs = d.get("conflict_score")
        primary = d.get("primary")
        cs_str = f"CS={cs:.3f}" if cs is not None else "CS=N/A"
        p_str = f"P={primary}" if primary else "P=N/A"
        logger.info(
            "  Step %2d | %-12s | H=%.3f | %s | %s | %s",
            i, d["action"], d["entropy"], cs_str, p_str, d["reason"][:60],
        )

    logger.info("=" * 60)
    logger.info("Smoke test PASSED if you see tokens above.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
