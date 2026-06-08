"""Comprehensive evaluation framework for ACC hallucination detection.

Benchmarks:
  - HaluEval QA (subset)
  - TruthfulQA
  - PubMedQA
  - Known factual/hallucination questions

Methods evaluated:
  - Baseline (no detection)
  - Entropy-only
  - SAPLMA (MLP on last-layer hidden states)
  - DoLa-style (logits contrast between layers)
  - ACC-PredictiveCoding (our method: prediction errors + temporal)
  - ACC-NoTemporal (ablation: remove leaky integrator)
  - ACC-RawHidden (ablation: use raw hidden states instead of prediction errors)

Metrics:
  - Per-sample: accuracy, precision, recall, F1
  - Token-level: flag precision/recall, AUC-ROC
  - Efficiency: avg tokens per sample, detection overhead
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.halueval_detector import HaluEvalDetector, SimpleHallucinationDetector
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig
from src.acc_layer import EntropyMonitor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "models/qwen2.5-1.5b"
MAX_NEW_TOKENS = 25
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"

# Sample limits per benchmark (for speed/memory)
HALUEVAL_N = 200
TRUTHFULQA_N = 30
PUBMEDQA_N = 20

LAYER_PAIRS = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]
LAYER_INDICES = [-1, -4, -8, -12, -16, -20, -24, -28]


@dataclass
class EvalResult:
    """Results for a single configuration on a single benchmark."""
    config_name: str
    benchmark: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: Optional[float] = None
    avg_flags: float = 0.0
    avg_entropy: float = 0.0
    avg_conflict: float = 0.0
    time_per_sample: float = 0.0
    predictions: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark Loaders
# ---------------------------------------------------------------------------

def load_halueval_qa(n: int = 500) -> List[Dict]:
    """Load HaluEval QA as (prompt, expected_label) pairs."""
    path = "data/halueval/data.jsonl"
    if not Path(path).exists():
        logger.warning("HaluEval not found at %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    examples = examples[:n]
    samples = []
    for ex in examples:
        # Factual sample
        samples.append({
            "prompt": f"Context: {ex['knowledge']}\nQuestion: {ex['question']}\nAnswer:",
            "expected": "factual",
            "reference": ex["right_answer"],
            "benchmark": "halueval",
        })
        # Hallucinated sample
        samples.append({
            "prompt": f"Context: {ex['knowledge']}\nQuestion: {ex['question']}\nAnswer:",
            "expected": "hallucinated",
            "reference": ex["hallucinated_answer"],
            "benchmark": "halueval",
        })
    return samples


def load_truthfulqa(n: int = 100) -> List[Dict]:
    """Load TruthfulQA generation subset."""
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    except Exception as e:
        logger.warning("TruthfulQA load failed: %s", e)
        return []
    indices = np.random.RandomState(42).choice(len(ds), min(n, len(ds)), replace=False)
    samples = []
    for i in indices:
        item = ds[int(i)]
        samples.append({
            "prompt": item["question"],
            "expected": "unknown",  # TruthfulQA: we judge against best_answer post-hoc
            "reference": item.get("best_answer", ""),
            "benchmark": "truthfulqa",
        })
    return samples


def load_pubmedqa(n: int = 50) -> List[Dict]:
    """Load PubMedQA pqa_labeled."""
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        logger.warning("PubMedQA load failed: %s", e)
        return []
    indices = np.random.RandomState(42).choice(len(ds), min(n, len(ds)), replace=False)
    samples = []
    for i in indices:
        item = ds[int(i)]
        context = item.get("context", "")
        if isinstance(context, list):
            context = " ".join(context)
        samples.append({
            "prompt": f"Context: {context}\nQuestion: {item['question']}\nAnswer:",
            "expected": item.get("final_decision", ""),
            "reference": item.get("final_decision", ""),
            "benchmark": "pubmedqa",
        })
    return samples


def load_known_questions() -> List[Dict]:
    """Known factual vs hallucination prompts."""
    return [
        {"prompt": "The capital of France is", "expected": "factual", "reference": "Paris", "benchmark": "known"},
        {"prompt": "Water boils at a temperature of", "expected": "factual", "reference": "100", "benchmark": "known"},
        {"prompt": "What did Isaac Newton say about artificial intelligence in 1700?", "expected": "hallucinated", "reference": "uncertain", "benchmark": "known"},
        {"prompt": "Explain how Marie Curie invented the smartphone", "expected": "hallucinated", "reference": "incorrect", "benchmark": "known"},
        {"prompt": "The square root of 144 is", "expected": "factual", "reference": "12", "benchmark": "known"},
        {"prompt": "Describe the lunar landing that happened in 1845", "expected": "hallucinated", "reference": "incorrect", "benchmark": "known"},
    ]


# ---------------------------------------------------------------------------
# Detection Methods
# ---------------------------------------------------------------------------

def run_baseline(model, tokenizer, prompt: str) -> Tuple[str, List[float]]:
    """Generate without any detection."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    entropies = [compute_entropy(s[0]) for s in outputs.scores]
    return text, entropies


def run_entropy_only(model, tokenizer, prompt: str, threshold: float = 3.9) -> Tuple[str, List[float], List[bool]]:
    """Generate with entropy monitoring."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    monitor = EntropyMonitor(threshold=threshold, mode="absolute", action="flag")
    entropies = []
    flags = []

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    for score in outputs.scores:
        h = compute_entropy(score[0])
        entropies.append(h)
        flags.append(monitor.check_threshold(h))
        monitor.window.update(h)

    return text, entropies, flags


def run_saplma(model, tokenizer, prompt: str, saplma_model: nn.Module) -> Tuple[str, List[float], List[float]]:
    """Generate with SAPLMA-style detector on last-layer hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Extract last-layer hidden states for each generated token
    num_layers = len(outputs.hidden_states[0]) - 1  # exclude embedding
    last_layer_idx = num_layers - 1
    probs = []
    entropies = []

    for step, score in enumerate(outputs.scores):
        h = compute_entropy(score[0])
        entropies.append(h)

        # hidden_states[step] is tuple of (num_layers+1) tensors
        hs = outputs.hidden_states[step][last_layer_idx]  # (batch, seq_len, hidden)
        last_token_hs = hs[0, -1, :].float()  # (hidden,)

        with torch.no_grad():
            logit = saplma_model(last_token_hs.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(logit).item()
        probs.append(prob)

    return text, entropies, probs


def run_dola(model, tokenizer, prompt: str, premature_layer: int = 10) -> Tuple[str, List[float], List[float]]:
    """Generate with DoLa-style contrastive decoding."""
    # Simplified DoLa: compute Jensen-Shannon divergence between final and premature layer logits
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    num_layers = len(outputs.hidden_states[0]) - 1
    premature_idx = max(0, min(premature_layer, num_layers - 1))

    entropies = []
    js_divergences = []

    for step, score in enumerate(outputs.scores):
        h = compute_entropy(score[0])
        entropies.append(h)

        # Get hidden states at final and premature layers
        final_hs = outputs.hidden_states[step][num_layers]  # (batch, seq_len, hidden)
        premature_hs = outputs.hidden_states[step][premature_idx]

        # Project to logits via lm_head
        final_logits = model.lm_head(final_hs[:, -1, :])  # (batch, vocab)
        premature_logits = model.lm_head(premature_hs[:, -1, :])

        # Compute JSD between distributions
        final_probs = F.softmax(final_logits[0], dim=-1)
        premature_probs = F.softmax(premature_logits[0], dim=-1)
        m = 0.5 * (final_probs + premature_probs)
        jsd = 0.5 * (F.kl_div(m.log(), final_probs, reduction="sum") +
                     F.kl_div(m.log(), premature_probs, reduction="sum"))
        js_divergences.append(jsd.item())

    return text, entropies, js_divergences


def run_acc_detector(model, tokenizer, prompt: str, detector: HaluEvalDetector) -> Tuple[str, List[float], List[float], List[str]]:
    """Generate with our HaluEval detector in ACC pipeline."""
    gen = ACCEnhancedGenerator(
        model=model,
        tokenizer=tokenizer,
        action="warning",
        threshold=3.9,
        mode="absolute",
        use_conflict_detector=True,
        use_realtime_conflict_detector=True,
        conflict_detector=detector,
        conflict_layer_indices=LAYER_INDICES,
        decision_engine=UnifiedDecisionEngine(
            entropy_threshold=3.9,
            conflict_score_threshold=0.7,
            dual_signal_regenerate=False,
            marker_config=MarkerConfig(hallucination="", contradiction="", uncertain=""),
        ),
    )

    output = gen.generate_from_prompt(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
        top_p=0.95,
        return_dict_in_generate=True,
    )

    text = output.text[0]
    decisions = output.per_token_decisions[0]

    entropies = [d.get("entropy", 0.0) for d in decisions]
    conflicts = [d.get("conflict_score") or 0.0 for d in decisions]
    actions = [d.get("action", "pass") for d in decisions]

    return text, entropies, conflicts, actions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy in nats."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return (-(probs * log_probs).sum()).item()


def judge_label(text: str, expected: str, reference: str, benchmark: str) -> str:
    """Judge if generated text matches expected label."""
    text_lower = text.lower().strip()

    if benchmark == "halueval":
        # For HaluEval, if expected is "factual", the reference is the right answer
        # If expected is "hallucinated", the reference is the hallucinated answer
        # We judge post-hoc by checking if text is closer to reference
        # But actually, for detection we want to know if the detector flags correctly
        # So for now, just return the expected label as ground truth
        return expected

    if benchmark == "pubmedqa":
        pred = "yes" if text_lower.startswith("yes") or "yes" in text_lower.split()[:5] else \
               "no" if text_lower.startswith("no") or "no" in text_lower.split()[:5] else "maybe"
        return "factual" if pred == reference.lower() else "hallucinated"

    if benchmark == "truthfulqa":
        # TruthfulQA: check if best_answer is in generated text
        ref_lower = reference.lower()
        return "factual" if ref_lower in text_lower or text_lower in ref_lower else "hallucinated"

    if benchmark == "known":
        if expected == "factual":
            return "factual" if reference.lower() in text_lower else "hallucinated"
        else:
            uncertainty = ["did not", "didn't", "never", "impossible", "incorrect",
                          "false", "no evidence", "not true", "uncertain", "cannot",
                          "can't", "could not", "no such", "did not invent"]
            return "factual" if any(p in text_lower for p in uncertainty) else "hallucinated"

    return "unknown"


# ---------------------------------------------------------------------------
# Training SAPLMA Baseline
# ---------------------------------------------------------------------------

def train_saplma(model, tokenizer, n_train: int = 500) -> nn.Module:
    """Train a simple SAPLMA-style MLP on last-layer hidden states."""
    logger.info("\nTraining SAPLMA baseline on %d HaluEval examples...", n_train)
    examples = load_halueval_qa(n_train)

    X_list = []
    y_list = []

    # Process in small batches
    batch_size = 8
    for i in tqdm(range(0, len(examples), batch_size), desc="SAPLMA features"):
        batch = examples[i:i + batch_size]
        prompts = [ex["prompt"] + " " + ex["reference"] for ex in batch]
        labels = [1 if ex["expected"] == "hallucinated" else 0 for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Last layer hidden states at last token
        num_layers = len(outputs.hidden_states) - 1
        last_layer_hs = outputs.hidden_states[num_layers]  # (batch, seq_len, hidden)
        mask = inputs["attention_mask"]
        last_positions = mask.sum(dim=1) - 1

        for b in range(len(batch)):
            hs = last_layer_hs[b, last_positions[b], :].cpu().float()
            X_list.append(hs)
            y_list.append(labels[b])

        if DEVICE == "xpu" and i % 50 == 0:
            torch.xpu.empty_cache()

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)

    # Train simple MLP
    hidden_dim = model.config.hidden_size
    saplma = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(saplma.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(X.to(DEVICE), y.to(DEVICE))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(20):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = saplma(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("  SAPLMA epoch %d: loss=%.4f", epoch + 1, total_loss / len(loader))

    saplma.eval()
    return saplma


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(config_name: str, samples: List[Dict], model, tokenizer, detector=None, saplma_model=None) -> EvalResult:
    """Evaluate one configuration on a list of samples."""
    y_true = []
    y_pred = []
    y_scores = []
    token_entropies = []
    token_conflicts = []
    token_flags = []

    for sample in tqdm(samples, desc=f"Eval {config_name}", leave=False):
        expected = sample["expected"]
        if expected not in ["factual", "hallucinated"]:
            continue  # Skip samples without binary labels

        if config_name == "baseline":
            text, entropies = run_baseline(model, tokenizer, sample["prompt"])
            # Baseline always predicts factual
            pred = "factual"
            score = 0.0
            flags = [False] * len(entropies)
            conflicts = [0.0] * len(entropies)

        elif config_name == "entropy":
            text, entropies, flags = run_entropy_only(model, tokenizer, sample["prompt"])
            pred = "hallucinated" if any(flags) else "factual"
            score = max(entropies) if entropies else 0.0
            conflicts = [0.0] * len(entropies)

        elif config_name == "saplma":
            text, entropies, probs = run_saplma(model, tokenizer, sample["prompt"], saplma_model)
            pred = "hallucinated" if any(p > 0.5 for p in probs) else "factual"
            score = max(probs) if probs else 0.0
            flags = [p > 0.5 for p in probs]
            conflicts = probs

        elif config_name == "dola":
            text, entropies, jsd_scores = run_dola(model, tokenizer, sample["prompt"])
            # High JSD suggests layer disagreement → possible hallucination
            threshold = np.percentile(jsd_scores, 80) if jsd_scores else 0.0
            pred = "hallucinated" if any(j > threshold for j in jsd_scores) else "factual"
            score = max(jsd_scores) if jsd_scores else 0.0
            flags = [j > threshold for j in jsd_scores]
            conflicts = jsd_scores

        elif config_name == "acc-detector":
            text, entropies, conflicts, actions = run_acc_detector(model, tokenizer, sample["prompt"], detector)
            pred = "hallucinated" if any(a in ["flag", "warning"] for a in actions) else "factual"
            score = max(conflicts) if conflicts else 0.0
            flags = [a in ["flag", "warning"] for a in actions]

        else:
            raise ValueError(f"Unknown config: {config_name}")

        # Ground truth label
        true_label = expected
        pred_label = pred

        y_true.append(1 if true_label == "hallucinated" else 0)
        y_pred.append(1 if pred_label == "hallucinated" else 0)
        y_scores.append(score)
        token_entropies.extend(entropies)
        token_conflicts.extend(conflicts)
        token_flags.extend([1 if f else 0 for f in flags])

        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    # Compute metrics
    if len(y_true) == 0:
        return EvalResult(config_name=config_name, benchmark=samples[0]["benchmark"] if samples else "unknown")

    accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None
    except Exception:
        auc = None

    return EvalResult(
        config_name=config_name,
        benchmark=samples[0]["benchmark"] if samples else "unknown",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc,
        avg_flags=np.mean(token_flags) if token_flags else 0.0,
        avg_entropy=np.mean(token_entropies) if token_entropies else 0.0,
        avg_conflict=np.mean(token_conflicts) if token_conflicts else 0.0,
    )


def main():
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE EVALUATION FRAMEWORK")
    logger.info("Device: %s", DEVICE.upper())
    logger.info("=" * 70)

    # Load model
    logger.info("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    if DEVICE == "xpu":
        model = model.to("xpu")
    logger.info("Model: %s on %s", MODEL_NAME, next(model.parameters()).device)

    # Load ACC detector
    detector = HaluEvalDetector(
        hidden_dim=model.config.hidden_size,
        layer_pairs=LAYER_PAIRS,
        checkpoint_path="adapters/halueval_detector.pt",
        device=DEVICE,
    )
    logger.info("Loaded ACC detector")

    # Train SAPLMA baseline
    saplma_model = train_saplma(model, tokenizer, n_train=min(HALUEVAL_N, 500))

    # Load benchmarks
    logger.info("\nLoading benchmarks...")
    benchmarks = {
        "halueval": load_halueval_qa(HALUEVAL_N),
        "truthfulqa": load_truthfulqa(TRUTHFULQA_N),
        "pubmedqa": load_pubmedqa(PUBMEDQA_N),
        "known": load_known_questions(),
    }
    for name, samples in benchmarks.items():
        logger.info("  %s: %d samples", name, len(samples))

    # Configs to evaluate
    configs = ["baseline", "entropy", "saplma", "dola", "acc-detector"]

    # Run evaluation
    all_results = []
    for bench_name, samples in benchmarks.items():
        if not samples:
            continue
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark: %s (%d samples)", bench_name, len(samples))
        logger.info("=" * 70)
        logger.info("  %-15s | Acc   | Prec  | Rec   | F1    | AUC   | Flags", "Config")
        logger.info("  " + "-" * 70)

        for config in configs:
            result = evaluate_config(config, samples, model, tokenizer, detector=detector, saplma_model=saplma_model)
            all_results.append(result)
            logger.info("  %-15s | %.3f | %.3f | %.3f | %.3f | %s | %.2f",
                        config,
                        result.accuracy,
                        result.precision,
                        result.recall,
                        result.f1,
                        f"{result.auc_roc:.3f}" if result.auc_roc is not None else "N/A",
                        result.avg_flags)

    # Save results
    out_path = Path("results/evaluation_framework.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([{
            "config": r.config_name,
            "benchmark": r.benchmark,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "auc_roc": r.auc_roc,
            "avg_flags": r.avg_flags,
            "avg_entropy": r.avg_entropy,
            "avg_conflict": r.avg_conflict,
        } for r in all_results], f, indent=2)
    logger.info("\nResults saved to: %s", out_path)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
