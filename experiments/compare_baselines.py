"""Compare generation strategies on the same prompt set.

Strategies:
  1. Base (no ACC)
  2. ACC-Entropy
  3. ACC-SelfConsistency
  4. ACC-ConflictDetector

Runs each strategy, saves outputs as JSONL, then calls evaluate_hallucination.py
for each. Produces a comparison table with statistical significance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import wilcoxon
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing src/ modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_integration import ACCEnhancedGenerator
from src.acc_conflict_detector import LatentConflictDetector
from experiments.evaluate_hallucination import token_level_f1


STRATEGIES = ["base", "acc_entropy", "acc_self_consistency", "acc_conflict_detector"]


def load_model_and_tokenizer(base_path: str, adapter_path: Optional[str] = None, device: str = "auto"):
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=False,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_base(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    token_probs = []
    if outputs.scores:
        for score in outputs.scores:
            probs = torch.softmax(score, dim=-1)
            token_probs.append(float(probs.max().item()))
    return {
        "prompt": prompt,
        "generated_text": text,
        "token_probs": token_probs,
    }


def generate_acc_entropy(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, threshold: float):
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        action="flag",
        mode="absolute",
    )
    out = gen.generate_from_prompt(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        return_dict_in_generate=True,
    )
    token_probs = []
    # Approximate token probs from entropy data (not available directly here)
    return {
        "prompt": prompt,
        "generated_text": out.text[0] if out.text else "",
        "per_token_entropy": out.per_token_entropy[0] if out.per_token_entropy else [],
    }


def generate_acc_self_consistency(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, threshold: float):
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        action="flag",
        mode="absolute",
        use_self_consistency=True,
        self_consistency_candidates=5,
        self_consistency_threshold=0.75,
    )
    out = gen.generate_from_prompt(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        return_dict_in_generate=True,
    )
    return {
        "prompt": prompt,
        "generated_text": out.text[0] if out.text else "",
        "consistency_score": out.consistency_score[0] if out.consistency_score else None,
        "contradiction_detected": out.contradiction_detected[0] if out.contradiction_detected else None,
    }


def generate_acc_conflict_detector(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, detector_path: str):
    # Load detector
    detector = LatentConflictDetector(hidden_dim=model.config.hidden_size)
    detector.load_state_dict(torch.load(Path(detector_path) / "detector.pt", map_location="cpu"))
    detector = detector.to(model.device)
    detector.eval()

    # Standard generation + detector on hidden states would require extractor integration
    # For baseline comparison, generate normally then run detector on hidden states
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Hidden states are not directly available from model.generate() with output_hidden_states
    # in all HF versions; skip detector scores for now and mark as N/A
    return {
        "prompt": prompt,
        "generated_text": text,
        "detector_conflict_score": None,
    }


def run_strategy(strategy: str, model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, cfg: Dict) -> Dict:
    if strategy == "base":
        return generate_base(model, tokenizer, prompt, max_new_tokens, temperature)
    elif strategy == "acc_entropy":
        return generate_acc_entropy(model, tokenizer, prompt, max_new_tokens, temperature, cfg["threshold"])
    elif strategy == "acc_self_consistency":
        return generate_acc_self_consistency(model, tokenizer, prompt, max_new_tokens, temperature, cfg["threshold"])
    elif strategy == "acc_conflict_detector":
        return generate_acc_conflict_detector(model, tokenizer, prompt, max_new_tokens, temperature, cfg.get("detector_path", "adapters/acc_conflict_detector"))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def statistical_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """Paired Wilcoxon signed-rank test."""
    # Filter out None values
    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a is not None and b is not None]
    if len(pairs) < 3:
        return {"test": "insufficient_data", "p_value": None, "significant_at_0.05": False}
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    stat, p = wilcoxon(a_vals, b_vals, alternative="two-sided")
    return {"test": "Wilcoxon signed-rank", "statistic": float(stat), "p_value": float(p), "significant_at_0.05": p < 0.05}


def main():
    parser = argparse.ArgumentParser(description="Compare baseline generation strategies")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML or JSON")
    parser.add_argument("--prompts", required=True, help="JSONL file with prompts and ground_truth_output")
    parser.add_argument("--output_dir", default="experiments/results/comparisons", help="Directory for outputs")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if cfg_path.suffix in (".yaml", ".yml"):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Load prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    # Load model once
    base_model = cfg["model"]["base_model"]
    adapter_path = cfg.get("adapter", {}).get("path")
    model, tokenizer = load_model_and_tokenizer(base_model, adapter_path, args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run each strategy
    strategy_results: Dict[str, List[Dict]] = {}
    for strategy in STRATEGIES:
        print(f"\n=== Running strategy: {strategy} ===")
        results = []
        for p in prompts:
            rec = run_strategy(strategy, model, tokenizer, p["prompt"], args.max_new_tokens, args.temperature, cfg.get("acc", {}))
            rec["ground_truth_output"] = p.get("ground_truth_output", "")
            rec["strategy"] = strategy
            results.append(rec)
        strategy_results[strategy] = results

        # Save JSONL
        jsonl_path = out_dir / f"{strategy}_outputs.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        # Evaluate
        eval_path = out_dir / f"{strategy}_eval.json"
        subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "experiments" / "evaluate_hallucination.py"),
             "--input", str(jsonl_path),
             "--output", str(eval_path)],
            check=True,
        )

    # Load evaluation summaries
    summaries = {}
    for strategy in STRATEGIES:
        eval_path = out_dir / f"{strategy}_eval.json"
        with open(eval_path, "r", encoding="utf-8") as f:
            summaries[strategy] = json.load(f)["aggregate"]

    # Comparison table
    comparison = []
    for strategy in STRATEGIES:
        s = summaries[strategy]
        comparison.append({
            "strategy": strategy,
            "hallucination_rate": s.get("hallucination_rate", None),
            "contradiction_rate": s.get("contradiction_rate", None),
            "mean_token_f1": s.get("mean_token_f1", None),
            "mean_calibration_error": s.get("mean_calibration_error", None),
        })

    # Pairwise statistical tests (vs base)
    base_results = strategy_results["base"]
    tests = {}
    for strategy in STRATEGIES[1:]:
        other = strategy_results[strategy]
        # Use lexical overlap as the scalar metric for paired testing
        base_overlaps = [token_level_f1(r["generated_text"], r.get("ground_truth_output", ""))["f1"] for r in base_results]
        other_overlaps = [token_level_f1(r["generated_text"], r.get("ground_truth_output", ""))["f1"] for r in other]
        tests[strategy] = statistical_test(base_overlaps, other_overlaps)

    final_report = {
        "comparison_table": comparison,
        "pairwise_tests_vs_base": tests,
        "config": cfg,
    }

    report_path = out_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for row in comparison:
        print(f"{row['strategy']:30s}  Halluc={row['hallucination_rate']:.3f}  F1={row['mean_token_f1']:.3f}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
