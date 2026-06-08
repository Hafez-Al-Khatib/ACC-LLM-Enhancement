"""Train PredictiveCodingDetector via self-consistency pseudo-labeling.

Pipeline:
  1. Generate N completions per prompt with diverse sampling
  2. Compute token-level consistency scores
  3. Label: high consistency -> factual, medium -> uncertain, low -> hallucination
  4. Train detector on hidden states with pseudo-labels
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from src.acc_conflict_detector import PredictiveCodingDetector, MultiLayerGenerationExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts designed to elicit different reliability profiles
# ---------------------------------------------------------------------------
FACTUAL_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The square root of 144 is",
    "The chemical formula for water is",
    "The speed of light is approximately",
    "Shakespeare wrote",
]

HALLUCINATION_PROMPTS = [
    "Tell me about the theory of quantum consciousness proposed by Einstein",
    "What did Isaac Newton say about artificial intelligence in 1700?",
    "Describe the discovery of the lost city of Atlantis by NASA",
    "Explain how Marie Curie invented the smartphone",
]

ALL_PROMPTS = FACTUAL_PROMPTS + HALLUCINATION_PROMPTS


def generate_with_extractor(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 12,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: str = "xpu",
    layer_indices=None,
) -> Tuple[str, List[Dict]]:
    """Generate text and return records with hidden states per step."""
    if layer_indices is None:
        layer_indices = [-1, -4, -8, -12, -16, -20, -24, -28]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    extractor = MultiLayerGenerationExtractor(model, layer_indices=layer_indices)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=LogitsProcessorList([extractor]),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    records = extractor.get_records(outputs.sequences, prompt_len=input_len)
    return text, records


def generate_diverse_completions(
    model,
    tokenizer,
    prompt: str,
    n_completions: int = 3,
    max_new_tokens: int = 12,
    device: str = "xpu",
) -> List[str]:
    """Generate N completions with diverse sampling."""
    configs = [
        {"temperature": 0.3, "top_p": 0.9},
        {"temperature": 0.8, "top_p": 0.95},
        {"temperature": 1.2, "top_p": 0.98},
    ]

    completions = []
    for i in range(n_completions):
        cfg = configs[i % len(configs)]
        text, _ = generate_with_extractor(
            model, tokenizer, prompt, max_new_tokens, **cfg, device=device
        )
        completions.append(text)

    return completions


def tokenize_completions(
    tokenizer,
    completions: List[str],
    prompt: str,
) -> List[List[int]]:
    """Tokenize completions and return token ID sequences (without prompt)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    token_sequences = []
    for comp in completions:
        full_text = prompt + comp
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        gen_ids = full_ids[prompt_len:]
        token_sequences.append(gen_ids)

    return token_sequences


def compute_consistency_labels(
    token_sequences: List[List[int]],
    max_len: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-position consistency scores and 3-way labels.

    Returns:
        consistency: (max_len,) float array, 0-1 score
        labels: (max_len,) int array, 0=factual, 1=uncertain, 2=hallucination
    """
    consistency = np.zeros(max_len, dtype=np.float32)
    labels = np.zeros(max_len, dtype=np.int64)

    for pos in range(max_len):
        tokens_at_pos = []
        for seq in token_sequences:
            if pos < len(seq):
                tokens_at_pos.append(seq[pos])

        if not tokens_at_pos:
            consistency[pos] = 0.0
            labels[pos] = 1
            continue

        counter = Counter(tokens_at_pos)
        most_common_count = counter.most_common(1)[0][1]
        consistency[pos] = most_common_count / len(tokens_at_pos)

        if consistency[pos] >= 0.7:
            labels[pos] = 0  # factual
        elif consistency[pos] >= 0.3:
            labels[pos] = 1  # uncertain
        else:
            labels[pos] = 2  # hallucination

    return consistency, labels


def build_training_data(
    model,
    tokenizer,
    prompts: List[str],
    n_completions: int = 3,
    max_new_tokens: int = 12,
    device: str = "xpu",
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    """
    Build training dataset from prompts.

    Returns:
        hidden_states_list: List of dicts, each mapping layer_idx -> tensor
        y_primary: (N,) long - 3-way labels
        y_secondary: (N,) long - 2-way labels
    """
    layer_indices = [-1, -4, -8, -12, -16, -20, -24, -28]
    hidden_states_list = []
    y_primary_list = []
    y_secondary_list = []

    logger.info("\nBuilding training data from %d prompts...", len(prompts))
    for prompt in tqdm(prompts, desc="Processing prompts"):
        # 1. Generate diverse completions for consistency labeling
        completions = generate_diverse_completions(
            model, tokenizer, prompt, n_completions, max_new_tokens, device
        )
        token_sequences = tokenize_completions(tokenizer, completions, prompt)
        consistency, labels = compute_consistency_labels(token_sequences, max_new_tokens)

        # 2. Extract hidden states from a single representative run
        _, records = generate_with_extractor(
            model, tokenizer, prompt, max_new_tokens, device=device
        )

        # 3. Match records with labels
        for step, record in enumerate(records):
            if step >= max_new_tokens:
                break

            # Convert hidden states to tensors
            hs_dict = {}
            for idx in layer_indices:
                if idx in record["hidden_states"]:
                    vec = torch.tensor(record["hidden_states"][idx], dtype=torch.float32)
                    hs_dict[idx] = vec

            if len(hs_dict) == len(layer_indices):
                hidden_states_list.append(hs_dict)
                y_primary_list.append(labels[step])
                y_secondary_list.append(0)  # no_contradiction default

        if device == "xpu":
            torch.xpu.empty_cache()

    y_primary = torch.tensor(y_primary_list, dtype=torch.long)
    y_secondary = torch.tensor(y_secondary_list, dtype=torch.long)

    logger.info("Dataset size: %d samples", len(hidden_states_list))
    for label_name, val in [("factual", 0), ("uncertain", 1), ("hallucination", 2)]:
        count = (y_primary == val).sum().item()
        logger.info("  %s: %d (%.1f%%)", label_name, count, 100 * count / len(y_primary))

    return hidden_states_list, y_primary, y_secondary


def collate_hidden_states(batch):
    """Collate list of hidden state dicts into batched dicts."""
    layer_indices = list(batch[0].keys())
    batched = {}
    for idx in layer_indices:
        stacked = torch.stack([item[idx] for item in batch])
        batched[idx] = stacked
    return batched


def train_detector(
    detector: PredictiveCodingDetector,
    hidden_states_list: List[Dict],
    y_primary: torch.Tensor,
    y_secondary: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "xpu",
) -> PredictiveCodingDetector:
    """Train detector on pseudo-labeled data."""
    detector = detector.to(device)
    detector.train()

    y_primary = y_primary.to(device)
    y_secondary = y_secondary.to(device)

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    logger.info("\nTraining detector for %d epochs...", epochs)
    for epoch in range(epochs):
        total_loss = 0.0
        correct_primary = 0
        total = 0

        # Manual batching
        indices = torch.randperm(len(hidden_states_list))
        for i in range(0, len(hidden_states_list), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_hs = [hidden_states_list[j] for j in batch_idx]
            yp_b = y_primary[batch_idx]
            ys_b = y_secondary[batch_idx]

            # Move to device and add batch dim
            batch_hs_device = {}
            for idx in batch_hs[0].keys():
                batch_hs_device[idx] = torch.stack([hs[idx] for hs in batch_hs]).to(device)

            optimizer.zero_grad()

            # No temporal state during training (each sample is independent)
            primary_logits, secondary_logits, conflict_score, _ = detector(batch_hs_device, prev_state=None)

            loss_primary = F.cross_entropy(primary_logits, yp_b)
            loss_secondary = F.cross_entropy(secondary_logits, ys_b)
            # Conflict score should correlate with uncertainty/hallucination
            target_conflicts = (yp_b > 0).float()
            loss_conflict = F.mse_loss(conflict_score.squeeze(), target_conflicts)

            loss = loss_primary + 0.5 * loss_secondary + 0.3 * loss_conflict
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            pred = primary_logits.argmax(dim=1)
            correct_primary += (pred == yp_b).sum().item()
            total += len(batch_idx)

        scheduler.step()
        acc = correct_primary / total if total > 0 else 0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  Epoch %d: loss=%.4f, primary_acc=%.3f, lr=%.2e",
                        epoch + 1, total_loss / total, acc, scheduler.get_last_lr()[0])

    detector.eval()
    logger.info("Training complete.")
    return detector


def save_detector(detector: PredictiveCodingDetector, path: str):
    """Save trained detector."""
    torch.save(detector.state_dict(), path)
    logger.info("Detector saved to: %s", path)


def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("DETECTOR TRAINING: Self-Consistency Pseudo-Labeling")
    logger.info("Device: %s", device.upper())
    logger.info("=" * 70)

    # Load model
    model_name = "models/qwen2.5-1.5b"
    logger.info("\nLoading model...")
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
    model.eval()
    logger.info("Model loaded on %s", next(model.parameters()).device)

    # Build detector
    hidden_dim = model.config.hidden_size
    layer_pairs = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]

    detector = PredictiveCodingDetector(
        hidden_dim=hidden_dim,
        layer_pairs=layer_pairs,
        temporal_decay=0.7,
    )
    logger.info("Detector initialized: %d params", sum(p.numel() for p in detector.parameters()))

    # Build training data
    hidden_states_list, y_primary, y_secondary = build_training_data(
        model, tokenizer, ALL_PROMPTS, n_completions=3, max_new_tokens=12, device=device
    )

    if len(hidden_states_list) == 0:
        logger.error("No training data generated!")
        return

    # Train
    detector = train_detector(
        detector, hidden_states_list, y_primary, y_secondary, epochs=50, batch_size=8, device=device
    )

    # Save
    save_path = "adapters/qwen2.5_detector.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_detector(detector, save_path)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
