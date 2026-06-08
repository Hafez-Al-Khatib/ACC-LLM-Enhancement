"""Real experiment: Qwen2.5-1.5B on Intel Arc XPU with full ACC pipeline.

Validates:
  - 1.5B model loads and runs inference on XPU
  - MultiLayerGenerationExtractor captures hidden states during generation
  - PredictiveCodingDetector produces real-time labels
  - ACCEnhancedGenerator makes intervention decisions
  - Per-token explainability works end-to-end
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig, ACCGenerationOutput
from src.acc_conflict_detector import PredictiveCodingDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


PROMPTS = [
    "The capital of France is",
    "In 1492, Christopher Columbus",
    "The theory of relativity was developed by",
    "Water boils at a temperature of",
]


def run_experiment(model_name: str = "models/qwen2.5-1.5b", max_new_tokens: int = 15):
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Qwen2.5-1.5B + ACC on %s", device.upper())
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading Qwen2.5-1.5B (1.5B params, hidden=1536, 28 layers)...")
    t0 = time.time()
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
    load_time = time.time() - t0
    logger.info("Loaded in %.1fs on %s", load_time, next(model.parameters()).device)

    # ------------------------------------------------------------------
    # 2. Build conflict detector for Qwen architecture
    # ------------------------------------------------------------------
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    logger.info("\n[2/5] Building PredictiveCodingDetector")
    logger.info("    hidden_dim=%d, num_layers=%d", hidden_dim, num_layers)

    # Qwen has 28 layers; tap deep hierarchy
    layer_indices = [-1, -4, -8, -12, -16, -20, -24, -28]
    layer_pairs = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]

    detector = PredictiveCodingDetector(
        hidden_dim=hidden_dim,
        layer_pairs=layer_pairs,
        temporal_decay=0.7,
    )
    # Load trained weights if available
    detector_path = "adapters/qwen2.5_detector.pt"
    if os.path.exists(detector_path):
        detector.load_state_dict(torch.load(detector_path, map_location=device))
        logger.info("    Loaded trained detector from: %s", detector_path)
    else:
        logger.info("    WARNING: No trained detector found, using random weights")
    detector.eval()
    if device == "xpu":
        detector = detector.to("xpu")
    logger.info("    Detector params: %d", sum(p.numel() for p in detector.parameters()))

    # ------------------------------------------------------------------
    # 3. Build ACC generator
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Building ACCEnhancedGenerator")
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="flag",
        threshold=1.5,
        mode="absolute",
        use_conflict_detector=True,
        use_realtime_conflict_detector=True,
        conflict_detector=detector,
        conflict_layer_indices=layer_indices,
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=1.5,
            conflict_score_threshold=0.6,
            dual_signal_regenerate=True,
            marker_config=MarkerConfig(
                hallucination=" [HALLUCINATION]",
                contradiction=" [CONTRADICTION]",
                uncertain=" [UNCERTAIN]",
            ),
        ),
    )

    # ------------------------------------------------------------------
    # 4. Run generation on each prompt
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Generating %d tokens per prompt...", max_new_tokens)
    results = []
    for prompt in PROMPTS:
        logger.info("\n  Prompt: '%s'", prompt)
        if device == "xpu":
            torch.xpu.empty_cache()

        t0 = time.time()
        output = gen.generate_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            return_dict_in_generate=True,
        )
        gen_time = time.time() - t0

        text = output.text[0]
        decisions = output.per_token_decisions[0]
        n_flags = sum(1 for d in decisions if d["action"] == "flag")
        n_regen = sum(1 for d in decisions if d["action"] == "regenerate")
        max_cs = max(
            (d.get("conflict_score") or 0.0 for d in decisions),
            default=0.0,
        )

        logger.info("    Output: %s", text.replace("\n", " "))
        logger.info("    Time: %.2fs | Flags: %d | Regens: %d | MaxConflict: %.3f",
                    gen_time, n_flags, n_regen, max_cs)

        results.append({
            "prompt": prompt,
            "output": text,
            "time_sec": gen_time,
            "n_flags": n_flags,
            "n_regenerations": n_regen,
            "max_conflict_score": max_cs,
            "decisions": decisions,
        })

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("[5/5] SUMMARY")
    logger.info("=" * 70)
    total_flags = sum(r["n_flags"] for r in results)
    total_regen = sum(r["n_regenerations"] for r in results)
    max_cs_overall = max(r["max_conflict_score"] for r in results)
    logger.info("Total prompts: %d", len(PROMPTS))
    logger.info("Total tokens generated: %d", len(PROMPTS) * max_new_tokens)
    logger.info("Total flags: %d", total_flags)
    logger.info("Total regenerations: %d", total_regen)
    logger.info("Max conflict score: %.3f", max_cs_overall)

    # Save raw results
    out_path = Path("results/xpu_experiment_qwen2.5-1.5b.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nRaw results saved to: %s", out_path)

    # Print one decision trace in full
    logger.info("\n--- Decision trace for last prompt ---")
    last_output = ACCGenerationOutput(
        sequences=output.sequences,
        text=output.text,
        per_token_entropy=output.per_token_entropy,
        uncertain_steps=output.uncertain_steps,
        per_token_decisions=output.per_token_decisions,
    )
    logger.info("\n" + gen.explain_decisions(last_output))

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_experiment()
