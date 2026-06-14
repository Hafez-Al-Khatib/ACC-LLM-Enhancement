#!/usr/bin/env python3
"""Publication-quality benchmark evaluation for ACC hallucination detection.

Combines real datasets, LLM-as-judge, strong baselines, and statistical testing.
Designed to run on RTX 4090 or similar GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.baselines import DoLaDetector, SAPLMADetector, EntropyDetector
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_halueval_samples(n: int = 100, seed: int = 42) -> List[Dict]:
    """Load HaluEval QA samples."""
    path = _PROJECT_ROOT / "data" / "halueval" / "data.jsonl"
    if not path.exists():
        logger.warning("HaluEval data not found at %s. Skipping.", path)
        return []

    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            right = item.get("right_answer", "")
            hallucinated = item.get("hallucinated_answer", "")
            samples.append({
                "prompt": item["question"],
                "expected": right,
                "hallucinated_answer": hallucinated,
                "type": "adversarial",
                "source": "halueval",
            })

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(samples), min(n, len(samples)), replace=False)
    return [samples[int(i)] for i in indices]


def load_truthfulqa_samples(n: int = 100, seed: int = 42) -> List[Dict]:
    """Load TruthfulQA multiple-choice samples."""
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), min(n, len(ds)), replace=False)
        samples = []
        for i in indices:
            item = ds[int(i)]
            samples.append({
                "prompt": item["question"],
                "expected": "truthful",
                "type": "factual",
                "source": "truthfulqa",
            })
        return samples
    except Exception as e:
        logger.warning("TruthfulQA load failed: %s", e)
        return []


def load_pubmedqa_samples(n: int = 50, seed: int = 42) -> List[Dict]:
    """Load PubMedQA samples."""
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), min(n, len(ds)), replace=False)
        samples = []
        for i in indices:
            item = ds[int(i)]
            ctx = item.get("context", "")
            if isinstance(ctx, list):
                ctx = " ".join(ctx)
            prompt = f"Context: {ctx}\nQuestion: {item['question']}\nAnswer:"
            samples.append({
                "prompt": prompt,
                "expected": item.get("final_decision", ""),
                "type": "factual",
                "source": "pubmedqa",
            })
        return samples
    except Exception as e:
        logger.warning("PubMedQA load failed: %s", e)
        return []


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, temperature: float = 0.8) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_with_uncertainty_prefix(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int) -> str:
    """Regenerate with an uncertainty-priming prefix."""
    prefix = "Wait, let me reconsider. I'm not entirely certain, but"
    full_prompt = f"{prompt}\n{prefix}"
    return prefix + " " + generate_baseline(model, tokenizer, full_prompt, max_new_tokens, device, seed, temperature=1.0)


def generate_entropy_intervention(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, detector: EntropyDetector) -> str:
    """Generate with entropy-based intervention."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    flagged = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = (-(probs * log_probs).sum()).item()
            if entropy > detector.threshold:
                flagged = True
                break
            probs_sample = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs_sample, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if flagged:
        return generate_with_uncertainty_prefix(model, tokenizer, prompt, max_new_tokens, device, seed + 999)
    return text


def generate_dola_intervention(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, detector: DoLaDetector) -> str:
    """Generate with DoLa-based intervention."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    flagged = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            mature_logits = outputs.logits[0, -1, :]
            premature_logits_list = []
            for layer_idx in detector.premature_layers:
                h = outputs.hidden_states[layer_idx][:, -1, :]
                logits = model.lm_head(h)[0]
                premature_logits_list.append(logits)
            premature_logits = torch.stack(premature_logits_list).mean(dim=0)
            p_prem = F.softmax(premature_logits, dim=-1)
            p_mat = F.softmax(mature_logits, dim=-1)
            m = 0.5 * (p_prem + p_mat)
            kl_pm = F.kl_div(m.log(), p_prem, reduction="sum")
            kl_mm = F.kl_div(m.log(), p_mat, reduction="sum")
            js_div = 0.5 * (kl_pm + kl_mm)
            if js_div.item() > detector.threshold:
                flagged = True
                break
            probs = F.softmax(mature_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if flagged:
        return generate_with_uncertainty_prefix(model, tokenizer, prompt, max_new_tokens, device, seed + 999)
    return text


def generate_saplma_intervention(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, detector: SAPLMADetector) -> str:
    """Generate with SAPLMA-based intervention."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    flagged = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().float()
            logit = detector.forward(last_hidden.to(detector.device))
            prob = torch.sigmoid(logit)
            if prob.item() > 0.5:
                flagged = True
                break
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if flagged:
        return generate_with_uncertainty_prefix(model, tokenizer, prompt, max_new_tokens, device, seed + 999)
    return text


def generate_selfcheckgpt_samples(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, n_samples: int = 5) -> List[str]:
    """Generate N samples for SelfCheckGPT consistency check."""
    texts = []
    for i in range(n_samples):
        text = generate_baseline(model, tokenizer, prompt, max_new_tokens, device, seed + i * 1000)
        texts.append(text)
    return texts


def selfcheckgpt_score(texts: List[str]) -> float:
    """Compute simple SelfCheckGPT score via n-gram overlap across samples.
    Lower score = more consistent = less likely hallucination.
    Returns consistency score in [0, 1]."""
    if len(texts) < 2:
        return 1.0

    def ngrams(text, n=2):
        words = text.lower().split()
        return set(zip(*[words[i:] for i in range(n)]))

    scores = []
    for i in range(len(texts)):
        others = [texts[j] for j in range(len(texts)) if j != i]
        ng_i = ngrams(texts[i])
        if not ng_i:
            continue
        overlaps = [len(ng_i & ngrams(other)) / len(ng_i) for other in others]
        scores.append(np.mean(overlaps))

    return np.mean(scores) if scores else 0.0


def llm_as_judge(judge_model, judge_tokenizer, prompt: str, response: str, expected: str, q_type: str, device: str) -> Dict:
    """Use a judge model to evaluate correctness.

    Returns dict with:
        correct: bool
        reason: short explanation
        label: one of [correct, incorrect, uncertain]
    """
    judge_prompt = f"""You are evaluating whether a language model response is correct, incorrect, or uncertain.

Question: {prompt}
Expected answer: {expected}
Model response: {response}

Classify the response as one of:
- correct: contains the expected answer and no false claims
- incorrect: contains false or fabricated information
- uncertain: appropriately expresses uncertainty or refusal

Respond with only one word: correct, incorrect, or uncertain. Then briefly explain why."""

    inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )
    text = judge_tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    clean = text.lower().strip()

    label = "uncertain"
    if clean.startswith("correct"):
        label = "correct"
    elif clean.startswith("incorrect"):
        label = "incorrect"
    elif clean.startswith("uncertain"):
        label = "uncertain"
    elif "incorrect" in clean.split()[:5]:
        label = "incorrect"
    elif "correct" in clean.split()[:5]:
        label = "correct"

    is_correct = label == "correct" or (q_type in ["hallucination", "uncertain"] and label == "uncertain")

    return {
        "correct": is_correct,
        "label": label,
        "reason": text.strip(),
    }


def paired_t_test(acc_a: List[bool], acc_b: List[bool]) -> Tuple[float, float]:
    """Paired t-test between two methods."""
    if len(acc_a) != len(acc_b) or len(acc_a) < 2:
        return 0.0, 1.0
    a = np.array(acc_a, dtype=float)
    b = np.array(acc_b, dtype=float)
    diff = a - b
    if np.std(diff) == 0:
        return 0.0, 1.0
    t_stat, p_value = stats.ttest_rel(a, b)
    return t_stat, p_value


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen_Qwen2.5-7B", help="Model to evaluate")
    parser.add_argument("--judge-model", default=None, help="Judge model (defaults to same as --model)")
    parser.add_argument("--halueval", type=int, default=100, help="Number of HaluEval samples")
    parser.add_argument("--truthfulqa", type=int, default=100, help="Number of TruthfulQA samples")
    parser.add_argument("--pubmedqa", type=int, default=50, help="Number of PubMedQA samples")
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/benchmark_evaluation.json")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM-as-judge for labels")
    parser.add_argument("--no-selfcheckgpt", action="store_true", help="Skip SelfCheckGPT baseline")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch, "xpu", None) and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logger.info("=" * 70)
    logger.info("BENCHMARK EVALUATION")
    logger.info("Model: %s | Device: %s", args.model, device)
    logger.info("=" * 70)

    # Load samples
    samples = []
    samples.extend(load_halueval_samples(args.halueval, args.seed))
    samples.extend(load_truthfulqa_samples(args.truthfulqa, args.seed + 1))
    samples.extend(load_pubmedqa_samples(args.pubmedqa, args.seed + 2))

    if not samples:
        logger.error("No samples loaded. Check dataset paths.")
        return

    logger.info("Loaded %d samples", len(samples))

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded on %s", next(model.parameters()).device)

    # Load judge model
    judge_model = None
    judge_tokenizer = None
    if args.use_llm_judge:
        judge_name = args.judge_model or args.model
        logger.info("Loading judge model: %s", judge_name)
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_name, trust_remote_code=True)
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Judge model loaded")

    # Initialize detectors
    logger.info("Initializing detectors...")
    dola = DoLaDetector(model, threshold=0.1, device=device)

    saplma = SAPLMADetector(hidden_dim=model.config.hidden_size, device=device)
    saplma.train_on_examples(
        model, tokenizer,
        factual_prompts=["The capital of Italy is", "The chemical symbol for oxygen is"],
        hallucinated_prompts=["How did Beethoven use machine learning to compose symphonies?",
                              "Explain how the ancient Egyptians built smartphones."],
        max_new_tokens=10, epochs=30,
    )

    entropy_det = EntropyDetector(threshold=3.9)

    custom_ckpt = _PROJECT_ROOT / "adapters" / "custom_detector.pt"
    acc_detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=[(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)],
        checkpoint_path=str(custom_ckpt) if custom_ckpt.exists() else None,
        device="cpu" if device == "cuda" else device,
    )
    acc_engine = ACCInterventionEngine(
        detector=acc_detector,
        conflict_threshold=0.5,
        relative_threshold=None,
        calibration_tokens=3,
        max_regenerations=1,
        temperature_bump=0.3,
        top_p_reduce=0.1,
    )

    # Results
    results = {
        "Baseline": [],
        "Entropy": [],
        "DoLa": [],
        "SAPLMA": [],
        "SelfCheckGPT": [],
        "ACC": [],
    }

    for i, sample in enumerate(samples, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]
        seed = args.seed + i

        logger.info("[%d/%d] %s", i, len(samples), prompt[:60])

        # Baseline
        base_text = generate_baseline(model, tokenizer, prompt, args.max_new_tokens, device, seed)
        results["Baseline"].append({"sample": sample, "text": base_text})

        # Entropy
        ent_text = generate_entropy_intervention(model, tokenizer, prompt, args.max_new_tokens, device, seed, entropy_det)
        results["Entropy"].append({"sample": sample, "text": ent_text})

        # DoLa
        dola_text = generate_dola_intervention(model, tokenizer, prompt, args.max_new_tokens, device, seed, dola)
        results["DoLa"].append({"sample": sample, "text": dola_text})

        # SAPLMA
        sap_text = generate_saplma_intervention(model, tokenizer, prompt, args.max_new_tokens, device, seed, saplma)
        results["SAPLMA"].append({"sample": sample, "text": sap_text})

        # SelfCheckGPT
        if not args.no_selfcheckgpt:
            sc_texts = generate_selfcheckgpt_samples(model, tokenizer, prompt, args.max_new_tokens, device, seed)
            results["SelfCheckGPT"].append({"sample": sample, "texts": sc_texts})

        # ACC
        acc_result = acc_engine.generate_with_logit_shift(
            model, tokenizer, prompt, args.max_new_tokens, 0.8, 0.95, device, seed
        )
        results["ACC"].append({"sample": sample, "text": acc_result["text"]})

        if device == "cuda" and i % 5 == 0:
            torch.cuda.empty_cache()

    # Labeling
    logger.info("\nLabeling responses...")
    for method_name, method_results in results.items():
        for item in method_results:
            sample = item["sample"]
            if method_name == "SelfCheckGPT":
                # SelfCheckGPT gets consistency score, not text label
                score = selfcheckgpt_score(item["texts"])
                # Convert consistency score to correctness: high consistency AND no hallucination markers
                # For simplicity, use heuristic: if consistency < 0.3, mark as incorrect/uncertain
                item["consistency"] = score
                item["correct"] = score > 0.3  # Simple threshold
            elif args.use_llm_judge and judge_model is not None:
                text = item["text"]
                judge_result = llm_as_judge(
                    judge_model, judge_tokenizer, sample["prompt"], text,
                    sample["expected"], sample["type"], device
                )
                item["correct"] = judge_result["correct"]
                item["judge_label"] = judge_result["label"]
                item["judge_reason"] = judge_result["reason"]
            else:
                # Fallback substring judge
                text = item["text"].lower()
                expected = sample["expected"].lower()
                hallucinated = sample.get("hallucinated_answer", "").lower()

                if sample["type"] == "factual":
                    item["correct"] = expected in text
                elif sample["type"] == "adversarial":
                    uncertainty_markers = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                                           "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                                           "no such", "not", "i don't know", "i'm not sure", "as an ai",
                                           "i am not sure", "i'm not certain", "unsure", "don't have enough information"]
                    has_uncertainty = any(m in text for m in uncertainty_markers)
                    has_right_answer = expected and expected in text
                    has_hallucinated = hallucinated and hallucinated in text
                    # Correct if it gives right answer or expresses uncertainty; incorrect if it hallucinates
                    item["correct"] = has_right_answer or (has_uncertainty and not has_hallucinated)
                else:
                    markers = ["did not", "didn't", "never", "impossible", "incorrect", "false",
                               "no evidence", "not true", "uncertain", "cannot", "can't", "could not",
                               "no such", "not", "i don't know", "i'm not sure", "as an ai"]
                    item["correct"] = any(m in text for m in markers)

    # Metrics
    summary = {}
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    baseline_corrects = [r["correct"] for r in results["Baseline"]]

    for method_name, method_results in results.items():
        if not method_results:
            continue
        corrects = [r["correct"] for r in method_results]
        accuracy = np.mean(corrects)
        lower, upper = bootstrap_ci(corrects)

        t_stat, p_value = paired_t_test(corrects, baseline_corrects)

        summary[method_name] = {
            "accuracy": float(accuracy),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "t_stat_vs_baseline": float(t_stat),
            "p_value_vs_baseline": float(p_value),
            "n": len(corrects),
        }

        sig = "*" if p_value < 0.05 else ""
        logger.info("%-15s | Acc: %.1f%% [%.1f, %.1f] | vs Baseline: p=%.3f%s",
                    method_name, accuracy * 100, lower * 100, upper * 100, p_value, sig)

    # Per-type
    logger.info("\nPer-type accuracy:")
    for q_type in ["factual", "hallucination", "adversarial"]:
        type_results_all = []
        for method_results in results.values():
            type_results_all.extend([r for r in method_results if r["sample"]["type"] == q_type])
        if not type_results_all:
            continue
        logger.info("\n  %s:", q_type)
        for method_name, method_results in results.items():
            if not method_results:
                continue
            type_results = [r for r in method_results if r["sample"]["type"] == q_type]
            if type_results:
                type_acc = np.mean([r["correct"] for r in type_results])
                logger.info("    %-15s: %.1f%% (n=%d)", method_name, type_acc * 100, len(type_results))
                summary[method_name][f"{q_type}_accuracy"] = float(type_acc)

    # Save detailed results
    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert booleans to Python bool for JSON serialization
    serializable_results = {}
    for method_name, method_results in results.items():
        serializable_results[method_name] = []
        for item in method_results:
            new_item = {k: bool(v) if isinstance(v, (np.bool_, torch.Tensor)) else v for k, v in item.items()}
            serializable_results[method_name].append(new_item)

    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "samples": [{"prompt": s["prompt"], "expected": s["expected"], "type": s["type"], "source": s["source"]} for s in samples],
            "results": serializable_results,
        }, f, indent=2)

    logger.info("\nSaved results to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
