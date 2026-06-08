"""Train a simple hallucination detector on HaluEval QA using teacher forcing.

For each HaluEval example, we extract hidden states in two conditions:
  - factual: knowledge + question + right_answer
  - hallucinated: knowledge + question + hallucinated_answer

A lightweight MLP is trained on the prediction-error features between layer pairs.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HALUEVAL_PATH = "data/halueval/data.jsonl"
MODEL_NAME = "models/qwen2.5-1.5b"
MAX_SAMPLES = 1000  # Use 2K examples = 4K hidden-state samples
LAYER_INDICES = [-1, -4, -8, -12, -16, -20, -24, -28]
LAYER_PAIRS = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3


def load_halueval(path: str, max_samples: int = None) -> List[Dict]:
    """Load HaluEval QA examples."""
    with open(path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    if max_samples:
        examples = examples[:max_samples]
    logger.info("Loaded %d HaluEval examples", len(examples))
    return examples


def build_prompt(knowledge: str, question: str, answer: str) -> str:
    """Build a prompt for teacher forcing."""
    return f"Context: {knowledge}\nQuestion: {question}\nAnswer: {answer}"


def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
) -> torch.Tensor:
    """
    Extract hidden states at the last token position for a batch of prompts.

    Returns:
        hidden_states: dict mapping layer_idx -> (batch, hidden_dim) tensor
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (num_layers+1) tensors
    # Each tensor: (batch, seq_len, hidden_dim)
    # We want the last non-padding token for each sequence
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

    hidden_dict = {}
    num_layers = len(outputs.hidden_states) - 1  # excluding embedding layer
    for idx in LAYER_INDICES:
        pos_idx = idx if idx >= 0 else num_layers + idx
        layer_hs = outputs.hidden_states[pos_idx]  # (batch, seq_len, hidden)

        # Gather last-token hidden state for each item
        batch_size = layer_hs.size(0)
        last_hs = layer_hs[torch.arange(batch_size), last_positions, :]  # (batch, hidden)
        hidden_dict[idx] = last_hs.cpu()

    return hidden_dict


def compute_prediction_errors(hidden_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Compute prediction-error features from layer pairs.

    Returns:
        features: (batch, hidden_dim) - average |h_l1 - h_l2| across pairs
    """
    errors = []
    for l1, l2 in LAYER_PAIRS:
        if l1 in hidden_dict and l2 in hidden_dict:
            error = torch.abs(hidden_dict[l1] - hidden_dict[l2])
            errors.append(error)

    if not errors:
        raise ValueError("No valid layer pairs found")

    return torch.stack(errors).mean(dim=0).float()  # (batch, hidden_dim)


class SimpleHallucinationDetector(nn.Module):
    """Lightweight MLP detector (SAPLMA-style)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


def build_dataset(
    model,
    tokenizer,
    examples: List[Dict],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build labeled dataset from HaluEval examples.

    Returns:
        X: (N, hidden_dim) prediction-error features
        y: (N,) binary labels, 0=factual, 1=hallucinated
    """
    features_list = []
    labels_list = []

    # Process in batches of prompt pairs
    batch_prompts = []
    batch_labels = []

    for ex in tqdm(examples, desc="Building dataset"):
        factual_prompt = build_prompt(ex["knowledge"], ex["question"], ex["right_answer"])
        hallucinated_prompt = build_prompt(ex["knowledge"], ex["question"], ex["hallucinated_answer"])

        batch_prompts.extend([factual_prompt, hallucinated_prompt])
        batch_labels.extend([0, 1])

        if len(batch_prompts) >= BATCH_SIZE * 2:
            hidden_dict = extract_hidden_states(model, tokenizer, batch_prompts, device)
            features = compute_prediction_errors(hidden_dict)
            features_list.append(features)
            labels_list.extend(batch_labels)
            batch_prompts = []
            batch_labels = []

    # Remainder
    if batch_prompts:
        hidden_dict = extract_hidden_states(model, tokenizer, batch_prompts, device)
        features = compute_prediction_errors(hidden_dict)
        features_list.append(features)
        labels_list.extend(batch_labels)

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.float32)

    logger.info("Dataset: %d samples | Factual: %d | Hallucinated: %d",
                len(y), (y == 0).sum().item(), (y == 1).sum().item())

    return X, y


def train(
    detector: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "xpu",
) -> nn.Module:
    """Train detector with binary cross-entropy."""
    detector = detector.to(device)

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    logger.info("\nTraining for %d epochs...", epochs)
    for epoch in range(epochs):
        detector.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = detector(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        scheduler.step()

        # Validation
        detector.eval()
        with torch.no_grad():
            val_logits = detector(X_val.to(device))
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val.to(device)).float().mean().item()

        train_acc = 0.0  # Skip for speed
        avg_loss = total_loss / len(train_dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Epoch %d: loss=%.4f | val_acc=%.3f | lr=%.2e",
                        epoch + 1, avg_loss, val_acc, scheduler.get_last_lr()[0])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = detector.state_dict().copy()

    logger.info("Best validation accuracy: %.3f", best_val_acc)

    if best_state is not None:
        detector.load_state_dict(best_state)

    return detector


def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("TRAINING DETECTOR ON HALUEVAL QA")
    logger.info("Device: %s", device.upper())
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
    if device == "xpu":
        model = model.to("xpu")
    model.eval()
    logger.info("Model loaded on %s", next(model.parameters()).device)

    # Load data
    examples = load_halueval(HALUEVAL_PATH, max_samples=MAX_SAMPLES)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    split = int(0.8 * len(examples))
    train_idx, val_idx = indices[:split], indices[split:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]

    logger.info("Train: %d | Val: %d", len(train_examples), len(val_examples))

    # Build datasets
    logger.info("\nExtracting training features...")
    X_train, y_train = build_dataset(model, tokenizer, train_examples, device)

    if device == "xpu":
        torch.xpu.empty_cache()

    logger.info("\nExtracting validation features...")
    X_val, y_val = build_dataset(model, tokenizer, val_examples, device)

    # Build detector
    hidden_dim = model.config.hidden_size
    detector = SimpleHallucinationDetector(input_dim=hidden_dim, hidden_dim=256)
    logger.info("\nDetector params: %d", sum(p.numel() for p in detector.parameters()))

    # Train
    detector = train(detector, X_train, y_train, X_val, y_val, epochs=EPOCHS, device=device)

    # Save
    save_dir = Path("adapters")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "halueval_detector.pt"
    torch.save(detector.state_dict(), save_path)
    logger.info("\nDetector saved to: %s", save_path)

    # Final metrics
    detector.eval()
    with torch.no_grad():
        val_logits = detector(X_val.to(device))
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()

        factual_probs = val_probs[y_val == 0].cpu().numpy()
        halluc_probs = val_probs[y_val == 1].cpu().numpy()

        logger.info("\nFinal validation:")
        logger.info("  Accuracy:  %.3f", (val_preds == y_val.to(device)).float().mean().item())
        logger.info("  Factual -   mean prob: %.3f", np.mean(factual_probs))
        logger.info("  Halluc  -   mean prob: %.3f", np.mean(halluc_probs))

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
