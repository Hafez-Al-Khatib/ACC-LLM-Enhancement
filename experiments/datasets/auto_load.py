"""Auto-download and format datasets for multi-vertical experiments.

Supports:
  - HuggingFace hub datasets (PubMedQA, FiQA, SciQ, etc.)
  - JSONL / CSV local files
  - Automatic train/val/test splitting
  - Consistent formatting to instruction-following schema.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertical dataset definitions (HuggingFace hub IDs + formatting)
# ---------------------------------------------------------------------------

VERTICAL_DATASETS = {
    "pubmedqa": {
        "hub_id": "pubmed_qa",
        "subset": "pqa_labeled",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "formatter": "pubmedqa",
    },
    "medmcqa": {
        "hub_id": "medmcqa",
        "subset": None,
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "formatter": "medmcqa",
    },
    "fiqa": {
        "hub_id": "google/fiqa",
        "subset": None,
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "formatter": "fiqa",
    },
    "financial_phrasebank": {
        "hub_id": "financial_phrasebank",
        "subset": "sentences_allagree",
        "splits": {"train": "train"},  # single split, we split manually
        "formatter": "sentiment",
    },
    "sciq": {
        "hub_id": "sciq",
        "subset": None,
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "formatter": "sciq",
    },
    "openbookqa": {
        "hub_id": "openbookqa",
        "subset": "main",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "formatter": "openbookqa",
    },
    "alpaca": {
        "hub_id": "tatsu-lab/alpaca",
        "subset": None,
        "splits": {"train": "train"},
        "formatter": "alpaca",
    },
    "dolly": {
        "hub_id": "databricks/databricks-dolly-15k",
        "subset": None,
        "splits": {"train": "train"},
        "formatter": "dolly",
    },
}


# ---------------------------------------------------------------------------
# Formatters: convert raw HF rows → {"instruction", "input", "output"}
# ---------------------------------------------------------------------------

def _format_pubmedqa(row):
    # PubMedQA: context + question → yes/no/maybe
    return {
        "instruction": "Answer the following medical question based on the provided context.",
        "input": f"Context: {row.get('context', row.get('pubmed', ''))}\nQuestion: {row.get('question', '')}",
        "output": row.get("final_decision", row.get("answer", "")),
    }


def _format_medmcqa(row):
    # MedMCQA: question + 4 options → correct answer letter
    options = [
        row.get("opa", ""),
        row.get("opb", ""),
        row.get("opc", ""),
        row.get("opd", ""),
    ]
    opts_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    answer_idx = row.get("cop", 0)  # correct option index (0-3)
    answer = chr(65 + int(answer_idx)) if isinstance(answer_idx, (int, float)) else "?"
    return {
        "instruction": "Select the correct answer to the following medical question.",
        "input": f"{row.get('question', '')}\n{opts_text}",
        "output": answer,
    }


def _format_fiqa(row):
    # FiQA: financial question → answer
    return {
        "instruction": "Answer the following financial question.",
        "input": row.get("question", ""),
        "output": row.get("answer", "") if isinstance(row.get("answer"), str) else "",
    }


def _format_sentiment(row):
    # Financial PhraseBank: sentence → sentiment label
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    label = row.get("label", row.get("sentiment", -1))
    text = label_map.get(int(label), "unknown") if isinstance(label, (int, float)) else str(label)
    return {
        "instruction": "Classify the sentiment of the following financial text as positive, negative, or neutral.",
        "input": row.get("sentence", ""),
        "output": text,
    }


def _format_sciq(row):
    # SciQ: question + 4 options → correct answer
    choices = row.get("choices", [])
    opts_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return {
        "instruction": "Answer the following science question.",
        "input": f"{row.get('question', '')}\n{opts_text}",
        "output": row.get("correct_answer", ""),
    }


def _format_openbookqa(row):
    # OpenBookQA: question + choices → correct choice text
    choices = row.get("choices", {}).get("text", [])
    labels = row.get("choices", {}).get("label", [])
    opts_text = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, choices))
    answer_key = row.get("answerKey", "")
    # Find the text corresponding to answerKey
    answer_text = ""
    for lbl, txt in zip(labels, choices):
        if lbl == answer_key:
            answer_text = txt
            break
    return {
        "instruction": "Answer the following question.",
        "input": f"{row.get('question_stem', row.get('question', ''))}\n{opts_text}",
        "output": answer_text,
    }


def _format_alpaca(row):
    # Alpaca already has instruction/input/output
    return {
        "instruction": row.get("instruction", ""),
        "input": row.get("input", ""),
        "output": row.get("output", row.get("response", "")),
    }


def _format_dolly(row):
    # Dolly: instruction + context → response
    return {
        "instruction": row.get("instruction", ""),
        "input": row.get("context", ""),
        "output": row.get("response", ""),
    }


_FORMATTERS = {
    "pubmedqa": _format_pubmedqa,
    "medmcqa": _format_medmcqa,
    "fiqa": _format_fiqa,
    "sentiment": _format_sentiment,
    "sciq": _format_sciq,
    "openbookqa": _format_openbookqa,
    "alpaca": _format_alpaca,
    "dolly": _format_dolly,
}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_and_format(
    dataset_key: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
):
    """Download a dataset from HF hub, format it, and save as JSONL.

    Parameters
    ----------
    dataset_key : str
        Key in VERTICAL_DATASETS (e.g., 'pubmedqa', 'fiqa', 'sciq').
    output_dir : str
        Directory to write train.jsonl, val.jsonl, test.jsonl.
    max_samples : int | None
        Cap total samples across all splits (useful for Jetson small-data runs).
    seed : int
        For reproducible subsampling.

    Returns
    -------
    dict with paths to written files and sample counts.
    """
    if dataset_key not in VERTICAL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key!r}. Available: {list(VERTICAL_DATASETS)}")

    meta = VERTICAL_DATASETS[dataset_key]
    formatter = _FORMATTERS[meta["formatter"]]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Download
    logger.info("Loading %s from HuggingFace...", meta["hub_id"])
    ds = load_dataset(meta["hub_id"], meta.get("subset"), trust_remote_code=True)

    # Map splits
    split_map = meta["splits"]
    splits = {}
    for role, hf_split in split_map.items():
        if hf_split in ds:
            splits[role] = ds[hf_split]
        else:
            # Some datasets only have 'train' — we split manually
            logger.warning("Split %s not found in %s", hf_split, meta["hub_id"])

    # If only train exists, do 80/10/10 split
    if set(splits.keys()) == {"train"} and len(splits["train"]) > 10:
        full = splits["train"].shuffle(seed=seed)
        n = len(full)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        splits = {
            "train": full.select(range(n_train)),
            "val": full.select(range(n_train, n_train + n_val)),
            "test": full.select(range(n_train + n_val, n)),
        }

    # Apply max_samples cap across all splits proportionally
    if max_samples is not None:
        total = sum(len(s) for s in splits.values())
        if total > max_samples:
            ratios = {k: len(v) / total for k, v in splits.items()}
            splits = {
                k: v.shuffle(seed=seed).select(range(int(max_samples * ratios[k])))
                for k, v in splits.items()
            }
            logger.info("Capped to %d total samples", max_samples)

    # Format and save
    result = {}
    for role, split_ds in splits.items():
        records = [formatter(row) for row in split_ds]
        # Add the unified "text" field for SFTTrainer
        for r in records:
            # Preserve ground-truth fields explicitly so eval scripts can compare generations
            r["instruction"] = r.get("instruction", "")
            r["input"] = r.get("input", "")
            r["output"] = r.get("output", "")
            r["text"] = (
                f"### Instruction:\n{r['instruction']}\n\n"
                f"### Input:\n{r['input']}\n\n"
                f"### Response:\n{r['output']}"
            )
        path = out / f"{role}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        result[role] = {"path": str(path), "count": len(records)}
        logger.info("Wrote %s: %d samples → %s", role, len(records), path)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download and format datasets for ACC LLM")
    parser.add_argument("--dataset", required=True, help=f"Dataset key. Choices: {', '.join(VERTICAL_DATASETS)}")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples (useful for Jetson)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_and_format(args.dataset, args.output_dir, args.max_samples, args.seed)
