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
from src.baselines import DoLaDetector, SAPLMADetector, EntropyDetector, SelfCheckGPTDetector
from src.halueval_detector import HaluEvalDetector
from src.acc_intervention import ACCInterventionEngine
from src.judge import get_judge

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


def generate_dola_contrastive(model, tokenizer, prompt: str, max_new_tokens: int, device: str, seed: int, detector: DoLaDetector) -> str:
    """Generate with DoLa contrastive decoding."""
    return DoLaDetector.generate_contrastive(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        seed=seed,
        premature_layers=detector.premature_layers,
        mature_layers=detector.mature_layers,
        alpha=0.1,
        temperature=0.8,
    )


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
            prob = detector.predict_from_hidden(last_hidden)
            if prob > 0.5:
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


def build_saplma_train_set(
    samples: List[Dict],
    n_per_class: int = 32,
    seed: int = 42,
    min_eval_total: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """Split ``samples`` into a SAPLMA training set and an evaluation set.

    The training set is balanced between factual prompts (label=0) and
    adversarial/hallucination-prone prompts (label=1). Samples selected for
    training are removed from the evaluation set to avoid data leakage.

    If the loaded samples are too few or are single-class (e.g. a tiny
    HaluEval-only run), an empty training set is returned and the caller
    should fall back to generic detector training examples.

    Returns:
        (train_samples, eval_samples)
    """
    rng = np.random.RandomState(seed)
    factual = [s for s in samples if s["type"] == "factual"]
    adversarial = [s for s in samples if s["type"] == "adversarial"]

    # Deterministic shuffle so repeated runs are reproducible.
    for subset in (factual, adversarial):
        rng.shuffle(subset)

    # We need both classes and enough total samples to leave a meaningful eval set.
    if len(factual) < 2 or len(adversarial) < 2 or len(samples) < min_eval_total + 4:
        logger.info(
            "Too few or single-class samples for benchmark-derived SAPLMA training (factual=%d, adversarial=%d, total=%d); using fallback.",
            len(factual), len(adversarial), len(samples),
        )
        return [], samples

    # Reserve up to n_per_class of each class while leaving a reasonable eval set.
    min_eval_per_class = max(1, min_eval_total // 2)
    train_factual_count = min(n_per_class, max(0, len(factual) - min_eval_per_class))
    train_adversarial_count = min(n_per_class, max(0, len(adversarial) - min_eval_per_class))

    # Ensure the overall eval set is not smaller than min_eval_total.
    while (len(factual) - train_factual_count + len(adversarial) - train_adversarial_count) < min_eval_total:
        if train_factual_count > train_adversarial_count:
            train_factual_count = max(0, train_factual_count - 1)
        else:
            train_adversarial_count = max(0, train_adversarial_count - 1)
        if train_factual_count == 0 and train_adversarial_count == 0:
            break

    if train_factual_count == 0 and train_adversarial_count == 0:
        return [], samples

    train_factual = factual[:train_factual_count]
    train_adversarial = adversarial[:train_adversarial_count]

    train_samples = []
    for s in train_factual:
        train_samples.append({"prompt": s["prompt"], "label": 0})
    for s in train_adversarial:
        train_samples.append({"prompt": s["prompt"], "label": 1})

    held_out_prompts = {s["prompt"] for s in train_samples}
    eval_samples = [s for s in samples if s["prompt"] not in held_out_prompts]

    logger.info(
        "SAPLMA training set: %d factual + %d adversarial = %d samples; evaluation: %d samples",
        len(train_factual), len(train_adversarial), len(train_samples), len(eval_samples),
    )
    return train_samples, eval_samples


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
    words = [w.strip(".,!?;:\"'()[]") for w in clean.split()[:10] if w.strip()]

    label = "uncertain"
    if not words:
        label = "uncertain"
    elif words[0] == "correct" or "correct" in words[:5]:
        label = "correct"
    elif words[0] == "incorrect" or "incorrect" in words[:5]:
        label = "incorrect"
    elif words[0] == "uncertain" or "uncertain" in words[:5]:
        label = "uncertain"

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
    parser.add_argument("--judge-type", default="local", choices=["local", "openai", "anthropic"],
                        help="Judge backend (local model or API)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI judge model")
    parser.add_argument("--anthropic-model", default="claude-3-5-sonnet-20241022", help="Anthropic judge model")
    parser.add_argument("--no-selfcheckgpt", action="store_true", help="Skip SelfCheckGPT baseline")
    parser.add_argument("--saplma-train-per-class", type=int, default=32,
                        help="Number of samples per class held out for SAPLMA training")
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

    # Load judge
    judge_fn = None
    if args.use_llm_judge:
        if args.judge_type == "local":
            judge_name = args.judge_model or args.model
            logger.info("Loading local judge model: %s", judge_name)
            judge_tokenizer = AutoTokenizer.from_pretrained(judge_name, trust_remote_code=True)
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
            judge_model = AutoModelForCausalLM.from_pretrained(
                judge_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Judge model loaded")
            judge_fn = get_judge("local", judge_model, judge_tokenizer, device)
        elif args.judge_type == "openai":
            logger.info("Using OpenAI judge: %s", args.openai_model)
            judge_fn = get_judge("openai")
            # Bind model name
            import functools
            from src.judge import openai_judge
            judge_fn = functools.partial(openai_judge, model=args.openai_model)
        elif args.judge_type == "anthropic":
            logger.info("Using Anthropic judge: %s", args.anthropic_model)
            import functools
            from src.judge import anthropic_judge
            judge_fn = functools.partial(anthropic_judge, model=args.anthropic_model)

    # Initialize detectors
    logger.info("Initializing detectors...")
    dola = DoLaDetector(model, threshold=0.1, device=device)

    saplma_train_samples, eval_samples = build_saplma_train_set(
        samples, n_per_class=args.saplma_train_per_class, seed=args.seed
    )

    saplma = SAPLMADetector(hidden_dim=model.config.hidden_size, device=device)
    if len(saplma_train_samples) >= 4:
        saplma.train_on_data(
            model, tokenizer, saplma_train_samples,
            epochs=30, lr=1e-3, val_split=0.2, batch_size=4,
        )
    else:
        logger.warning("Too few samples for SAPLMA train/val split; using tiny fallback examples.")
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

    selfcheck_detector = SelfCheckGPTDetector(
        model=model,
        tokenizer=tokenizer,
        n_samples=5,
        temperature=0.8,
        device=device,
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

    for i, sample in enumerate(eval_samples, 1):
        prompt = sample["prompt"]
        expected = sample["expected"]
        q_type = sample["type"]
        seed = args.seed + i

        logger.info("[%d/%d] %s", i, len(eval_samples), prompt[:60])

        # Baseline
        base_text = generate_baseline(model, tokenizer, prompt, args.max_new_tokens, device, seed)
        results["Baseline"].append({"sample": sample, "text": base_text})

        # Entropy
        ent_text = generate_entropy_intervention(model, tokenizer, prompt, args.max_new_tokens, device, seed, entropy_det)
        results["Entropy"].append({"sample": sample, "text": ent_text})

        # DoLa (contrastive decoding)
        dola_text = generate_dola_contrastive(model, tokenizer, prompt, args.max_new_tokens, device, seed, dola)
        results["DoLa"].append({"sample": sample, "text": dola_text})

        # SAPLMA
        sap_text = generate_saplma_intervention(model, tokenizer, prompt, args.max_new_tokens, device, seed, saplma)
        results["SAPLMA"].append({"sample": sample, "text": sap_text})

        # SelfCheckGPT
        if not args.no_selfcheckgpt:
            sc_result = selfcheck_detector.detect_sequence(
                model, tokenizer, prompt, args.max_new_tokens, device, seed
            )
            results["SelfCheckGPT"].append({"sample": sample, "selfcheck": sc_result})

        # ACC
        acc_result = acc_engine.generate_with_logit_shift(
            model, tokenizer, prompt, args.max_new_tokens, 0.8, 0.95, device, seed
        )
        results["ACC"].append({"sample": sample, "text": acc_result["text"]})

        if device == "cuda" and i % 5 == 0:
            torch.cuda.empty_cache()
        elif device == "xpu" and i % 5 == 0:
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

    # Labeling
    logger.info("\nLabeling responses...")
    for method_name, method_results in results.items():
        for item in method_results:
            sample = item["sample"]
            if method_name == "SelfCheckGPT":
                # SelfCheckGPT gets consistency score, not text label
                score = item["selfcheck"]["consistency"]
                # Convert consistency score to correctness: high consistency AND no hallucination markers
                # For simplicity, use heuristic: if consistency < 0.3, mark as incorrect/uncertain
                item["consistency"] = score
                item["correct"] = score > 0.3  # Simple threshold
            elif args.use_llm_judge and judge_fn is not None:
                text = item["text"]
                judge_result = judge_fn(
                    sample["prompt"], text, sample["expected"], sample["type"]
                )
                item["correct"] = judge_result["correct"]
                item["judge_label"] = judge_result["label"]
                item["judge_reason"] = judge_result["reason"]
                item["judge_backend"] = judge_result.get("judge_type", "unknown")
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
            new_item = {}
            for k, v in item.items():
                if isinstance(v, (np.bool_, torch.Tensor)):
                    new_item[k] = bool(v)
                elif isinstance(v, np.floating):
                    new_item[k] = float(v)
                elif isinstance(v, np.integer):
                    new_item[k] = int(v)
                elif isinstance(v, np.ndarray):
                    new_item[k] = v.tolist()
                else:
                    new_item[k] = v
            serializable_results[method_name].append(new_item)

    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "samples": [{"prompt": s["prompt"], "expected": s["expected"], "type": s["type"], "source": s["source"]} for s in eval_samples],
            "results": serializable_results,
        }, f, indent=2)

    logger.info("\nSaved results to: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
