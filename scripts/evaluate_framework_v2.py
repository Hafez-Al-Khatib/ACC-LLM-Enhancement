"""Lightweight but rigorous evaluation framework.

Methods: baseline, entropy, SAPLMA, ACC-detector
Benchmarks: HaluEval (200), TruthfulQA (30), PubMedQA (20), Known (6)
Metrics: accuracy, precision, recall, f1, AUC-ROC, avg flags
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.halueval_detector import HaluEvalDetector
from src.acc_integration import ACCEnhancedGenerator, UnifiedDecisionEngine, MarkerConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models/qwen2.5-1.5b"
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
MAX_NEW_TOKENS = 20


def compute_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return (-(probs * log_probs).sum()).item()


def load_benchmarks() -> Dict[str, List[Dict]]:
    benchmarks = {}

    # HaluEval
    path = "data/halueval/data.jsonl"
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f][:200]
        samples = []
        for ex in examples:
            prompt = f"Context: {ex['knowledge']}\nQuestion: {ex['question']}\nAnswer:"
            samples.append({"prompt": prompt, "label": "factual", "ref": ex["right_answer"], "bench": "halueval"})
            samples.append({"prompt": prompt, "label": "hallucinated", "ref": ex["hallucinated_answer"], "bench": "halueval"})
        benchmarks["halueval"] = samples
        logger.info("HaluEval: %d samples", len(samples))

    # TruthfulQA
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        indices = np.random.RandomState(42).choice(len(ds), min(30, len(ds)), replace=False)
        benchmarks["truthfulqa"] = [
            {"prompt": ds[int(i)]["question"], "label": "unknown", "ref": ds[int(i)].get("best_answer", ""), "bench": "truthfulqa"}
            for i in indices
        ]
        logger.info("TruthfulQA: %d samples", len(benchmarks["truthfulqa"]))
    except Exception as e:
        logger.warning("TruthfulQA: %s", e)

    # PubMedQA
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        indices = np.random.RandomState(42).choice(len(ds), min(20, len(ds)), replace=False)
        samples = []
        for i in indices:
            item = ds[int(i)]
            ctx = item.get("context", "")
            if isinstance(ctx, list):
                ctx = " ".join(ctx)
            samples.append({
                "prompt": f"Context: {ctx}\nQuestion: {item['question']}\nAnswer:",
                "label": item.get("final_decision", ""),
                "ref": item.get("final_decision", ""),
                "bench": "pubmedqa",
            })
        benchmarks["pubmedqa"] = samples
        logger.info("PubMedQA: %d samples", len(samples))
    except Exception as e:
        logger.warning("PubMedQA: %s", e)

    # Known
    benchmarks["known"] = [
        {"prompt": "The capital of France is", "label": "factual", "ref": "Paris", "bench": "known"},
        {"prompt": "Water boils at a temperature of", "label": "factual", "ref": "100", "bench": "known"},
        {"prompt": "What did Isaac Newton say about AI in 1700?", "label": "hallucinated", "ref": "uncertain", "bench": "known"},
        {"prompt": "Explain how Marie Curie invented the smartphone", "label": "hallucinated", "ref": "incorrect", "bench": "known"},
        {"prompt": "The square root of 144 is", "label": "factual", "ref": "12", "bench": "known"},
        {"prompt": "Describe the lunar landing in 1845", "label": "hallucinated", "ref": "incorrect", "bench": "known"},
    ]
    logger.info("Known: %d samples", len(benchmarks["known"]))

    return benchmarks


def train_saplma(model, tokenizer, examples: List[Dict]) -> nn.Module:
    """Train SAPLMA on last-layer hidden states."""
    logger.info("Training SAPLMA on %d examples...", len(examples))
    X, y = [], []

    batch_size = 4
    for i in tqdm(range(0, len(examples), batch_size), desc="SAPLMA train", leave=False):
        batch = examples[i:i + batch_size]
        texts = [ex["prompt"] + " " + ex["ref"] for ex in batch]
        labels = [1 if ex["label"] == "hallucinated" else 0 for ex in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        n_layers = len(outputs.hidden_states) - 1
        hs = outputs.hidden_states[n_layers]  # (batch, seq, hidden)
        mask = inputs["attention_mask"]
        last_pos = mask.sum(dim=1) - 1

        for b in range(len(batch)):
            X.append(hs[b, last_pos[b], :].cpu().float())
            y.append(labels[b])

        if DEVICE == "xpu" and i % 50 == 0:
            torch.xpu.empty_cache()

    X = torch.stack(X).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    saplma = nn.Sequential(
        nn.Linear(model.config.hidden_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    ).to(DEVICE)

    opt = torch.optim.Adam(saplma.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(15):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = saplma(xb).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("  Epoch %d: loss=%.4f", epoch + 1, total / len(loader))

    saplma.eval()
    return saplma


def evaluate_method(method: str, samples: List[Dict], model, tokenizer, detector=None, saplma=None) -> Dict:
    """Evaluate a single method."""
    y_true, y_pred, y_scores = [], [], []
    n_flags = []

    for sample in tqdm(samples, desc=method, leave=False):
        label = sample["label"]
        if label not in ["factual", "hallucinated"]:
            continue

        prompt = sample["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        if method == "baseline":
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                     temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
            pred = "hallucinated"  # Baseline never detects; we use this as null predictor
            score = 0.0
            flags = 0

        elif method == "entropy":
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                     temperature=0.8, top_p=0.95, return_dict_in_generate=True,
                                     output_scores=True, pad_token_id=tokenizer.eos_token_id)
            entropies = [compute_entropy(s[0]) for s in out.scores]
            pred = "hallucinated" if any(h > 3.9 for h in entropies) else "factual"
            score = max(entropies) if entropies else 0.0
            flags = sum(1 for h in entropies if h > 3.9)

        elif method == "saplma":
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                     temperature=0.8, top_p=0.95, return_dict_in_generate=True,
                                     output_scores=True, output_hidden_states=True,
                                     pad_token_id=tokenizer.eos_token_id)
            n_layers = len(out.hidden_states[0]) - 1
            probs = []
            for step in range(len(out.scores)):
                hs = out.hidden_states[step][n_layers][0, -1, :].float()
                p = torch.sigmoid(saplma(hs.unsqueeze(0).to(DEVICE))).item()
                probs.append(p)
            pred = "hallucinated" if any(p > 0.5 for p in probs) else "factual"
            score = max(probs) if probs else 0.0
            flags = sum(1 for p in probs if p > 0.5)

        elif method == "acc":
            gen = ACCEnhancedGenerator(
                model=model, tokenizer=tokenizer, action="warning", threshold=3.9, mode="absolute",
                use_conflict_detector=True, use_realtime_conflict_detector=True,
                conflict_detector=detector, conflict_layer_indices=[-1, -4, -8, -12, -16, -20, -24, -28],
                decision_engine=UnifiedDecisionEngine(
                    entropy_threshold=3.9, conflict_score_threshold=0.7,
                    dual_signal_regenerate=False,
                    marker_config=MarkerConfig(hallucination="", contradiction="", uncertain=""),
                ),
            )
            output = gen.generate_from_prompt(prompt, max_new_tokens=MAX_NEW_TOKENS,
                                              temperature=0.8, top_p=0.95, return_dict_in_generate=True)
            actions = [d.get("action", "pass") for d in output.per_token_decisions[0]]
            conflicts = [d.get("conflict_score") or 0.0 for d in output.per_token_decisions[0]]
            pred = "hallucinated" if any(a in ["flag", "warning"] for a in actions) else "factual"
            score = max(conflicts) if conflicts else 0.0
            flags = sum(1 for a in actions if a in ["flag", "warning"])

        y_true.append(1 if label == "hallucinated" else 0)
        y_pred.append(1 if pred == "hallucinated" else 0)
        y_scores.append(score)
        n_flags.append(flags)

        if DEVICE == "xpu":
            torch.xpu.empty_cache()

    if len(y_true) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": None, "flags": 0}

    acc = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None
    except Exception:
        auc = None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
        "flags": float(np.mean(n_flags)),
    }


def main():
    logger.info("=" * 70)
    logger.info("EVALUATION FRAMEWORK V2")
    logger.info("=" * 70)

    logger.info("\nLoading model...")
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
        device=DEVICE,
    )

    benchmarks = load_benchmarks()

    # Train SAPLMA on subset
    halueval_train = benchmarks.get("halueval", [])[:200]
    saplma = train_saplma(model, tokenizer, halueval_train)

    methods = ["baseline", "entropy", "saplma", "acc"]
    all_results = []

    for bench_name, samples in benchmarks.items():
        if not samples:
            continue
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark: %s (%d samples)", bench_name, len(samples))
        logger.info("=" * 70)
        logger.info("  %-12s | Acc   | Prec  | Rec   | F1    | AUC   | Flags", "Method")
        logger.info("  " + "-" * 65)

        for method in methods:
            result = evaluate_method(method, samples, model, tokenizer, detector=detector, saplma=saplma)
            all_results.append({"benchmark": bench_name, "method": method, **result})
            logger.info("  %-12s | %.3f | %.3f | %.3f | %.3f | %s | %.2f",
                        method, result["accuracy"], result["precision"], result["recall"],
                        result["f1"], f"{result['auc']:.3f}" if result["auc"] is not None else "N/A",
                        result["flags"])

    # Save
    out = Path("results/evaluation_framework_v2.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nSaved to: %s", out)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
