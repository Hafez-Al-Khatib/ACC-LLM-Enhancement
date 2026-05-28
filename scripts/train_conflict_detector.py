"""Train Approach B: Latent-State Conflict Detector (generation-time).

Reads per-token JSONL data produced by ``generate_conflict_data.py``,
splits it into train/val, and trains a :class:`LatentConflictDetector`
with early stopping and per-class metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_conflict_detector import LatentConflictDetector


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """PyTorch Dataset over (hidden_state, label) records."""

    def __init__(self, records: List[Dict], label_map: Dict[str, int]):
        self.X = torch.tensor([r["hidden_state"] for r in records], dtype=torch.float32)
        self.y = torch.tensor([label_map[r["label"]] for r in records], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_checkpoint(
    detector: LatentConflictDetector,
    label_map: Dict[str, int],
    save_dir: Path,
    metrics: Dict,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(detector.state_dict(), save_dir / "detector.pt")

    config = {
        "hidden_dim": detector.hidden_dim,
        "num_labels": len(detector.LABELS),
        "labels": detector.LABELS,
        "label_map": label_map,
        "metrics": metrics,
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    detector: LatentConflictDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    detector.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = detector(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.inference_mode()
def evaluate(
    detector: LatentConflictDetector,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: List[str],
) -> Tuple[float, Dict, float]:
    detector.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = detector(X)
        loss = criterion(logits, y)

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return avg_loss, report, macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the generation-time conflict detector")
    parser.add_argument(
        "--data",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "acc_training" / "generated_conflict_data.jsonl"),
        help="Path to JSONL file with per-token records",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(_PROJECT_ROOT / "adapters" / "acc_conflict_detector"),
        help="Directory to save the best checkpoint",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension (auto-detected from data if omitted)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (disabled if omitted)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="conflict-detector",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # WandB setup
    # ------------------------------------------------------------------
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if k not in ("wandb_project", "wandb_run_name")},
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.data} ...")
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Run scripts/generate_conflict_data.py first."
        )

    records = load_records(args.data)
    print(f"Loaded {len(records)} token records.")

    if len(records) == 0:
        raise ValueError("No records found in data file.")

    label_names = LatentConflictDetector.LABELS
    label_map = {name: idx for idx, name in enumerate(label_names)}

    # Validate labels
    bad_labels = [r["label"] for r in records if r["label"] not in label_map]
    if bad_labels:
        raise ValueError(f"Unknown labels found: {set(bad_labels)}")

    # Shuffle and split
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    split_idx = int(len(records) * (1 - args.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    # Auto-detect hidden_dim if not provided
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = len(records[0]["hidden_state"])
        print(f"Auto-detected hidden_dim: {hidden_dim}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_ds = TokenDataset(train_records, label_map)
    val_ds = TokenDataset(val_records, label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    detector = LatentConflictDetector(
        hidden_dim=hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    detector = detector.to(device)
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"LatentConflictDetector: {n_params/1e3:.1f}K parameters")

    optimizer = torch.optim.AdamW(
        detector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_state: Dict | None = None

    print(f"\n{'Epoch':>6s} {'TrainLoss':>10s} {'ValLoss':>9s} {'MacroF1':>8s}")
    print("-" * 40)

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(detector, train_loader, optimizer, criterion, device)
        val_loss, report, macro_f1 = evaluate(detector, val_loader, criterion, device, label_names)

        scheduler.step(macro_f1)

        print(
            f"{epoch:6d} {train_loss:10.4f} {val_loss:9.4f} {macro_f1:8.4f}"
        )

        # Per-class metrics
        per_class_metrics = {}
        for name in label_names:
            p = report[name]["precision"]
            r = report[name]["recall"]
            f1 = report[name]["f1-score"]
            support = int(report[name]["support"])
            print(f"  {name:12s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={support}")
            per_class_metrics[name] = {"precision": p, "recall": r, "f1": f1, "support": support}

        if args.wandb_project:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "macro_f1": macro_f1,
                **{f"{k}_f1": v["f1"] for k, v in per_class_metrics.items()},
            })

        # Checkpointing
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            best_state = detector.state_dict()
            save_checkpoint(
                detector,
                label_map,
                Path(args.save_dir),
                metrics={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "macro_f1": macro_f1,
                    "per_class": {
                        name: {
                            "precision": report[name]["precision"],
                            "recall": report[name]["recall"],
                            "f1": report[name]["f1-score"],
                            "support": int(report[name]["support"]),
                        }
                        for name in label_names
                    },
                },
            )
            print(f"  [*] New best model saved (macro_f1={macro_f1:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    if args.wandb_project:
        import wandb
        wandb.summary["best_macro_f1"] = best_macro_f1
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Training complete. Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoint saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
