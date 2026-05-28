"""End-to-end medical QA benchmark.

Loads PubMedQA, runs generation with/without ACC, and evaluates using
experiments/evaluate_hallucination.py.

Expected dataset layout (from experiments/datasets/auto_load.py):
  experiments/datasets/pubmedqa/{train,val,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_integration import ACCEnhancedGenerator


def load_model_and_adapter(base_path: str, adapter_path: Optional[str], device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=False,
    )
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_prompts(test_jsonl: str, max_samples: Optional[int] = None) -> List[Dict]:
    records = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Build prompt from instruction/input/output fields if available
            prompt = rec.get("text", "")
            gt = rec.get("output", "")
            if not prompt:
                inst = rec.get("instruction", "")
                inp = rec.get("input", "")
                out = rec.get("output", "")
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
                gt = out
            records.append({"prompt": prompt, "ground_truth_output": gt})
    if max_samples:
        records = records[:max_samples]
    return records


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


def generate_acc(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, acc_cfg: Dict) -> Dict:
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        threshold=float(acc_cfg.get("threshold", 3.5)),
        action=acc_cfg.get("action", "flag"),
        mode=acc_cfg.get("mode", "absolute"),
        use_self_consistency=acc_cfg.get("use_self_consistency", False),
        self_consistency_candidates=int(acc_cfg.get("self_consistency_candidates", 5)),
        self_consistency_threshold=float(acc_cfg.get("self_consistency_threshold", 0.75)),
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
        "per_token_entropy": out.per_token_entropy[0] if out.per_token_entropy else [],
        "consistency_score": out.consistency_score[0] if out.consistency_score else None,
        "contradiction_detected": out.contradiction_detected[0] if out.contradiction_detected else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Medical QA benchmark")
    parser.add_argument("--config", required=True, help="Experiment config YAML/JSON")
    parser.add_argument("--dataset", default="experiments/datasets/pubmedqa/test.jsonl")
    parser.add_argument("--output_dir", default="experiments/results/medical_qa")
    parser.add_argument("--max_samples", type=int, default=None)
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

    base_model = cfg["model"]["base_model"]
    adapter_path = cfg.get("adapter", {}).get("path")
    acc_cfg = cfg.get("acc", {})

    model, tokenizer = load_model_and_adapter(base_model, adapter_path, args.device)
    prompts = load_prompts(args.dataset, args.max_samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Base generation ----
    print("\n=== Running Base Generation ===")
    base_results = []
    for p in prompts:
        rec = generate_base(model, tokenizer, p["prompt"], args.max_new_tokens, args.temperature)
        rec["ground_truth_output"] = p["ground_truth_output"]
        base_results.append(rec)

    base_jsonl = out_dir / "base_outputs.jsonl"
    with open(base_jsonl, "w", encoding="utf-8") as f:
        for rec in base_results:
            f.write(json.dumps(rec) + "\n")

    base_eval = out_dir / "base_eval.json"
    subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "experiments" / "evaluate_hallucination.py"),
         "--input", str(base_jsonl),
         "--output", str(base_eval)],
        check=True,
    )

    # ---- ACC generation ----
    print("\n=== Running ACC Generation ===")
    acc_results = []
    for p in prompts:
        rec = generate_acc(model, tokenizer, p["prompt"], args.max_new_tokens, args.temperature, acc_cfg)
        rec["ground_truth_output"] = p["ground_truth_output"]
        acc_results.append(rec)

    acc_jsonl = out_dir / "acc_outputs.jsonl"
    with open(acc_jsonl, "w", encoding="utf-8") as f:
        for rec in acc_results:
            f.write(json.dumps(rec) + "\n")

    acc_eval = out_dir / "acc_eval.json"
    subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "experiments" / "evaluate_hallucination.py"),
         "--input", str(acc_jsonl),
         "--output", str(acc_eval)],
        check=True,
    )

    # ---- Load and compare ----
    with open(base_eval, "r", encoding="utf-8") as f:
        base_agg = json.load(f)["aggregate"]
    with open(acc_eval, "r", encoding="utf-8") as f:
        acc_agg = json.load(f)["aggregate"]

    report = {
        "base": base_agg,
        "acc": acc_agg,
        "config": cfg,
    }
    report_path = out_dir / "medical_qa_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("MEDICAL QA BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30s} {'Base':>10s} {'ACC':>10s}")
    print("-" * 60)
    for key in ["hallucination_rate", "contradiction_rate", "mean_token_f1", "mean_calibration_error"]:
        b = base_agg.get(key)
        a = acc_agg.get(key)
        b_str = f"{b:.3f}" if b is not None else "N/A"
        a_str = f"{a:.3f}" if a is not None else "N/A"
        print(f"{key:<30s} {b_str:>10s} {a_str:>10s}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
