"""Train detector on model-specific data.

Uses prediction-error features between layer pairs, same as before.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/detector_training_data.json"
OUTPUT_PATH = "adapters/custom_detector.pt"
LAYER_PAIRS = [(-28, -24), (-24, -20), (-20, -16), (-16, -12), (-12, -8), (-8, -4), (-4, -1)]
HIDDEN_DIM = 1536  # Qwen2.5-1.5B
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path: str):
    with open(path) as f:
        data = json.load(f)

    features = []
    labels = []

    for item in data:
        hidden_states = item["hidden_states"]
        hs_dict = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in hidden_states.items()}

        # Compute prediction errors between layer pairs
        errors = []
        for l1, l2 in LAYER_PAIRS:
            if l1 in hs_dict and l2 in hs_dict:
                error = torch.abs(hs_dict[l1] - hs_dict[l2])
                errors.append(error)

        if errors:
            feature = torch.stack(errors).mean(dim=0)
            features.append(feature)
            labels.append(item["label"])

    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y


class DetectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train():
    logger.info("Loading data from %s...", DATA_PATH)
    X, y = load_data(DATA_PATH)
    logger.info("Loaded %d samples | Features: %d | Positives: %d | Negatives: %d",
                len(X), X.shape[1], (y == 1).sum().item(), (y == 0).sum().item())

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DetectorMLP(input_dim=HIDDEN_DIM, hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d | Loss: %.4f | Val Acc: %.2f%%", epoch + 1, avg_loss, val_acc * 100)

    # Save best
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), OUTPUT_PATH)
    logger.info("\nBest validation accuracy: %.2f%%", best_val_acc * 100)
    logger.info("Saved detector to %s", OUTPUT_PATH)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(DEVICE))
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs > 0.5).float()

        tp = ((val_preds == 1) & (y_val.to(DEVICE) == 1)).sum().item()
        fp = ((val_preds == 1) & (y_val.to(DEVICE) == 0)).sum().item()
        tn = ((val_preds == 0) & (y_val.to(DEVICE) == 0)).sum().item()
        fn = ((val_preds == 0) & (y_val.to(DEVICE) == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logger.info("Precision: %.2f | Recall: %.2f | F1: %.2f", precision, recall, f1)


if __name__ == "__main__":
    train()
