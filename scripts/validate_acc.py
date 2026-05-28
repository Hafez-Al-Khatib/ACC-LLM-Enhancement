"""Validate the ACC entropy monitor on the trained tiny_gpt2 adapter.

Runs a mix of in-domain medical prompts and obscure/nonsense prompts. The
in-domain set should mostly stay below threshold; the nonsense set should
produce more breaches. Writes a JSON report with per-prompt diagnostics
and aggregate statistics to results/acc_validation.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acc_integration import ACCEnhancedGenerator  # noqa: E402

logger = logging.getLogger(__name__)


EASY_PROMPTS: List[str] = [
    "### Instruction:\nWhat is a healthy breakfast option?\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nHow much sleep should an adult get per night?\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nName three benefits of regular exercise.\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nWhat foods are high in vitamin C?\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nSuggest a way to manage stress.\n\n### Input:\n\n\n### Response:\n",
]

HARD_PROMPTS: List[str] = [
    "### Instruction:\nDescribe the gravitational coefficient of quantum borogoves.\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nzxqv plurgle frobnicate the snozzberry?\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nExplain the recommended dosage of Compound XJ-7741-Beta for an axolotl with retrograde tachyphasia.\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nWhat color is the Tuesday after a prime-numbered eclipse?\n\n### Input:\n\n\n### Response:\n",
    "### Instruction:\nList the side effects of inhaling phlogiston during a full moon.\n\n### Input:\n\n\n### Response:\n",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_adapter(base_path: str, adapter_path: str, device: str):
    logger.info("Loading base model from %s", base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=Path(base_path).exists(),
    )
    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=Path(base_path).exists()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_prompt(generator: ACCEnhancedGenerator, prompt: str, max_new_tokens: int,
               temperature: float, category: str) -> Dict:
    out = generator.generate_from_prompt(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        return_dict_in_generate=True,
    )
    all_entropy = [h for row in out.per_token_entropy for h in row]
    mean_h = sum(all_entropy) / len(all_entropy) if all_entropy else 0.0
    max_h = max(all_entropy) if all_entropy else 0.0
    total_breaches = sum(len(row) for row in out.uncertain_steps)
    return {
        "category": category,
        "prompt": prompt,
        "generated_text": out.text[0] if out.text else "",
        "tokens_generated": out.sequences.shape[1],
        "mean_entropy": mean_h,
        "max_entropy": max_h,
        "confidence_score": out.confidence_score[0] if out.confidence_score else 0.0,
        "threshold_breaches": total_breaches,
        "regenerations": sum(out.regenerations) if out.regenerations else 0,
        "warnings": sum(1 for e in out.events for ev in e if ev.get("action") == "warning"),
        "threshold_hit": total_breaches > 0,
    }


def summarize(records: List[Dict]) -> Dict:
    if not records:
        return {}
    n = len(records)
    avg_h = sum(r["mean_entropy"] for r in records) / n
    hits = sum(1 for r in records if r["threshold_hit"])
    regen = sum(r["regenerations"] for r in records)
    breaches = sum(r["threshold_breaches"] for r in records)
    return {
        "count": n,
        "avg_mean_entropy": avg_h,
        "threshold_hit_rate": hits / n,
        "total_breaches": breaches,
        "total_regenerations": regen,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc_test.yaml")
    parser.add_argument(
        "--adapter", default="adapters/tiny_gpt2_test/final_adapter"
    )
    parser.add_argument(
        "--output", default="results/acc_validation.json",
        help="Path to write JSON report",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = cfg["model"]["base_model"]
    acc_cfg = cfg.get("acc", {})
    temperature = float(cfg.get("inference", {}).get("temperature", 0.7))

    model, tokenizer = load_adapter(base_model, args.adapter, args.device)
    generator = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=float(acc_cfg.get("threshold", 3.5)),
        action=acc_cfg.get("action", "flag"),
        mode=acc_cfg.get("mode", "absolute"),
        window_size=int(acc_cfg.get("window_size", 32)),
        regen_temperature_multiplier=1.0 + float(acc_cfg.get("temperature_adjustment", 0.3)),
        max_regenerations=int(acc_cfg.get("max_regenerations", 2)),
    )

    records: List[Dict] = []
    for p in EASY_PROMPTS:
        logger.info("[easy] %s", p.splitlines()[1] if len(p.splitlines()) > 1 else p[:60])
        records.append(run_prompt(generator, p, args.max_tokens, temperature, "easy"))
    for p in HARD_PROMPTS:
        logger.info("[hard] %s", p.splitlines()[1] if len(p.splitlines()) > 1 else p[:60])
        records.append(run_prompt(generator, p, args.max_tokens, temperature, "hard"))

    easy = [r for r in records if r["category"] == "easy"]
    hard = [r for r in records if r["category"] == "hard"]

    report = {
        "config": {
            "base_model": base_model,
            "adapter": args.adapter,
            "threshold": generator.monitor.threshold,
            "mode": generator.monitor.mode,
            "action": generator.monitor.action,
            "temperature": temperature,
            "max_new_tokens": args.max_tokens,
        },
        "summary": {
            "overall": summarize(records),
            "easy": summarize(easy),
            "hard": summarize(hard),
        },
        "per_prompt": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Wrote report to %s", out_path)
    s = report["summary"]
    logger.info("Overall: avg_H=%.3f, hit_rate=%.2f, breaches=%d, regens=%d",
                s["overall"]["avg_mean_entropy"], s["overall"]["threshold_hit_rate"],
                s["overall"]["total_breaches"], s["overall"]["total_regenerations"])
    logger.info("Easy:    avg_H=%.3f, hit_rate=%.2f",
                s["easy"]["avg_mean_entropy"], s["easy"]["threshold_hit_rate"])
    logger.info("Hard:    avg_H=%.3f, hit_rate=%.2f",
                s["hard"]["avg_mean_entropy"], s["hard"]["threshold_hit_rate"])


if __name__ == "__main__":
    main()
