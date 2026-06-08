"""Train Approach B: Predictive Coding Conflict Detector (generation-time).

Reads per-token JSONL data produced by ``generate_conflict_data.py``,
splits it into train/val, and trains a :class:`PredictiveCodingDetector`
with hierarchical classification (primary 3-way + secondary 2-way).

Supports both **hard labels** (standard classification) and **soft labels**
(probability vectors from pseudo-labeling pipelines).

Label mapping from old 4-way to new hierarchical:
    supported     -> primary=supported,     secondary=None
    hallucinated  -> primary=unsupported,   secondary=hallucinated
    uncertain     -> primary=uncertain,      secondary=None
    contradictory -> primary=unsupported,   secondary=contradictory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_conflict_detector import PredictiveCodingDetector


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

OLD_TO_PRIMARY = {
    "supported": "supported",
    "hallucinated": "unsupported",
    "uncertain": "uncertain",
    "contradictory": "unsupported",
}

OLD_TO_SECONDARY = {
    "supported": None,
    "hallucinated": "hallucinated",
    "uncertain": None,
    "contradictory": "contradictory",
}

PRIMARY_IDX_MAP = {label: i for i, label in enumerate(PredictiveCodingDetector.PRIMARY_LABELS)}
SECONDARY_IDX_MAP = {label: i for i, label in enumerate(PredictiveCodingDetector.SECONDARY_LABELS)}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Down-weights easy examples and focuses on hard negatives.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (batch, num_classes) raw logits.
            target: (batch,) class indices.

        Returns:
            Scalar loss.
        """
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy with soft (probability) targets.

    Args:
        logits: (batch, num_classes) raw logits.
        targets: (batch, num_classes) probability distributions.
        weight: Optional (num_classes,) per-class weights.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if weight is not None:
        # Weight by the target class with highest probability
        max_class = targets.argmax(dim=-1)
        sample_weights = weight[max_class]
        loss = loss * sample_weights
    return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiLayerTokenDataset(Dataset):
    """PyTorch Dataset over (multi_layer_hidden_states, primary_label, secondary_label) records.

    Supports both hard labels (string class names) and soft labels (probability vectors).
    Soft labels are read from record fields ``primary_soft`` and ``secondary_soft``
    if present; otherwise hard labels are converted to one-hot vectors.
    """

    def __init__(self, records: List[Dict], use_soft_labels: bool = False):
        self.samples = []
        self.use_soft_labels = use_soft_labels

        for r in records:
            old_label = r["label"]
            primary = OLD_TO_PRIMARY[old_label]
            secondary = OLD_TO_SECONDARY[old_label]

            # Convert hidden_states dict to tensor dict
            hidden_states = {}
            for layer_idx, vec in r.get("hidden_states", {}).items():
                key = int(layer_idx)
                hidden_states[key] = torch.tensor(vec, dtype=torch.float32)

            # Fallback for old format: single hidden_state
            if not hidden_states and "hidden_state" in r:
                hidden_states = {-1: torch.tensor(r["hidden_state"], dtype=torch.float32)}

            sample = {
                "hidden_states": hidden_states,
                "primary": primary,
                "secondary": secondary,
            }

            # Soft labels (if available and requested)
            if use_soft_labels:
                primary_soft = r.get("primary_soft")
                secondary_soft = r.get("secondary_soft")

                if primary_soft is not None:
                    sample["primary_soft"] = torch.tensor(primary_soft, dtype=torch.float32)
                else:
                    # Convert hard label to one-hot
                    ph = torch.zeros(len(PredictiveCodingDetector.PRIMARY_LABELS))
                    ph[PRIMARY_IDX_MAP[primary]] = 1.0
                    sample["primary_soft"] = ph

                if secondary_soft is not None:
                    sample["secondary_soft"] = torch.tensor(secondary_soft, dtype=torch.float32)
                elif secondary is not None:
                    sh = torch.zeros(len(PredictiveCodingDetector.SECONDARY_LABELS))
                    sh[SECONDARY_IDX_MAP[secondary]] = 1.0
                    sample["secondary_soft"] = sh
                else:
                    sample["secondary_soft"] = None

            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate for variable hidden-state dicts."""
    result = {
        "hidden_states": [b["hidden_states"] for b in batch],
        "primary": [b["primary"] for b in batch],
        "secondary": [b["secondary"] for b in batch],
    }
    if "primary_soft" in batch[0]:
        result["primary_soft"] = torch.stack([b["primary_soft"] for b in batch])
    if "secondary_soft" in batch[0]:
        secondary_softs = [b["secondary_soft"] for b in batch]
        result["secondary_soft"] = secondary_softs  # List[Tensor | None]
    return result


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
    detector: PredictiveCodingDetector,
    save_dir: Path,
    metrics: Dict,
    suffix: str = "",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f"detector{suffix}.pt"
    torch.save(detector.state_dict(), save_dir / name)

    config = {
        "hidden_dim": detector.hidden_dim,
        "layer_pairs": detector.layer_pairs,
        "temporal_decay": detector.temporal_decay,
        "primary_labels": detector.PRIMARY_LABELS,
        "secondary_labels": detector.SECONDARY_LABELS,
        "metrics": metrics,
    }
    with open(save_dir / f"config{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def compute_loss(
    detector: PredictiveCodingDetector,
    batch: Dict,
    primary_criterion: nn.Module,
    secondary_criterion: nn.Module,
    device: torch.device,
    secondary_weight: float = 0.5,
    use_soft_labels: bool = False,
    conflict_score_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute primary + secondary + optional auxiliary loss for a batch.

    Args:
        detector: The model.
        batch: Collated batch dict.
        primary_criterion: Loss function for primary head.
        secondary_criterion: Loss function for secondary head.
        device: torch device.
        secondary_weight: Weight for secondary loss relative to primary.
        use_soft_labels: If True, use soft targets (probability vectors).
        conflict_score_weight: Weight for auxiliary conflict score MSE loss.
            If > 0, trains the conflict score head to predict the entropy
            of the primary distribution as a proxy for uncertainty.

    Returns:
        total_loss: Scalar tensor.
        metrics: Dict of loss components.
    """
    batch_size = len(batch["primary"])

    # Prepare labels
    primary_indices = []
    secondary_indices = []
    secondary_mask = []

    for p_label, s_label in zip(batch["primary"], batch["secondary"]):
        primary_indices.append(PRIMARY_IDX_MAP[p_label])
        if s_label is not None:
            secondary_indices.append(SECONDARY_IDX_MAP[s_label])
            secondary_mask.append(1)
        else:
            secondary_indices.append(0)
            secondary_mask.append(0)

    primary_labels_t = torch.tensor(primary_indices, dtype=torch.long, device=device)
    secondary_labels_t = torch.tensor(secondary_indices, dtype=torch.long, device=device)
    secondary_mask_t = torch.tensor(secondary_mask, dtype=torch.float32, device=device)

    # Forward each sample independently
    all_primary_logits = []
    all_secondary_logits = []
    all_conflict_scores = []

    for hidden_states in batch["hidden_states"]:
        hs = {k: v.to(device).unsqueeze(0) for k, v in hidden_states.items()}
        primary_logits, secondary_logits, conflict_score, _ = detector(hs, prev_state=None)
        all_primary_logits.append(primary_logits)
        all_secondary_logits.append(secondary_logits)
        all_conflict_scores.append(conflict_score)

    primary_logits = torch.cat(all_primary_logits, dim=0)  # (batch, 3)
    secondary_logits = torch.cat(all_secondary_logits, dim=0)  # (batch, 2)
    conflict_scores = torch.cat(all_conflict_scores, dim=0)  # (batch, 1)

    # Primary loss
    if use_soft_labels and "primary_soft" in batch:
        primary_soft = batch["primary_soft"].to(device)  # (batch, 3)
        primary_loss = soft_cross_entropy(primary_logits, primary_soft)
    else:
        primary_loss = primary_criterion(primary_logits, primary_labels_t)

    # Secondary loss
    if use_soft_labels and "secondary_soft" in batch:
        secondary_soft_list = batch["secondary_soft"]
        secondary_losses = []
        for i in range(batch_size):
            if secondary_soft_list[i] is not None and secondary_mask[i] == 1:
                soft = secondary_soft_list[i].to(device).unsqueeze(0)  # (1, 2)
                logit = secondary_logits[i:i+1]  # (1, 2)
                secondary_losses.append(soft_cross_entropy(logit, soft))
        if secondary_losses:
            secondary_loss = torch.stack(secondary_losses).mean()
        else:
            secondary_loss = torch.tensor(0.0, device=device)
    else:
        raw_secondary_loss = secondary_criterion(secondary_logits, secondary_labels_t)
        # Safe division: if no secondary labels in batch, loss is 0
        denom = secondary_mask_t.sum()
        if denom > 0:
            secondary_loss = (raw_secondary_loss * secondary_mask_t).sum() / denom
        else:
            secondary_loss = torch.tensor(0.0, device=device)

    # Auxiliary conflict score loss
    # Train conflict_score to approximate the entropy of primary distribution
    # High entropy (uncertain) -> high conflict score
    if conflict_score_weight > 0.0:
        primary_probs = F.softmax(primary_logits, dim=-1)
        entropy = -(primary_probs * torch.log(primary_probs + 1e-8)).sum(dim=-1, keepdim=True)  # (batch, 1)
        # Normalize entropy to [0, 1] roughly (max entropy for 3 classes is log(3) ~ 1.1)
        normalized_entropy = entropy / np.log(3)
        conflict_score_loss = F.mse_loss(conflict_scores, normalized_entropy)
    else:
        conflict_score_loss = torch.tensor(0.0, device=device)

    total_loss = primary_loss + secondary_weight * secondary_loss + conflict_score_weight * conflict_score_loss

    metrics = {
        "primary_loss": primary_loss.item(),
        "secondary_loss": secondary_loss.item(),
        "conflict_score_loss": conflict_score_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


def train_one_epoch(
    detector: PredictiveCodingDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    primary_criterion: nn.Module,
    secondary_criterion: nn.Module,
    device: torch.device,
    secondary_weight: float = 0.5,
    use_soft_labels: bool = False,
    conflict_score_weight: float = 0.0,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    detector.train()
    total_primary_loss = 0.0
    total_secondary_loss = 0.0
    total_conflict_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        loss, metrics = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, device,
            secondary_weight, use_soft_labels, conflict_score_weight
        )
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(detector.parameters(), grad_clip)

        optimizer.step()

        batch_size = len(batch["primary"])
        total_primary_loss += metrics["primary_loss"] * batch_size
        total_secondary_loss += metrics["secondary_loss"] * batch_size
        total_conflict_loss += metrics["conflict_score_loss"] * batch_size
        total_samples += batch_size
        pbar.set_postfix(
            loss=f"{metrics['total_loss']:.4f}",
            p=f"{metrics['primary_loss']:.3f}",
            s=f"{metrics['secondary_loss']:.3f}",
        )

    return {
        "primary_loss": total_primary_loss / total_samples,
        "secondary_loss": total_secondary_loss / total_samples,
        "conflict_score_loss": total_conflict_loss / total_samples,
        "total_loss": (total_primary_loss + total_secondary_loss + total_conflict_loss) / total_samples,
    }


@torch.inference_mode()
def evaluate(
    detector: PredictiveCodingDetector,
    dataloader: DataLoader,
    primary_criterion: nn.Module,
    secondary_criterion: nn.Module,
    device: torch.device,
    secondary_weight: float = 0.5,
    use_soft_labels: bool = False,
    conflict_score_weight: float = 0.0,
) -> Tuple[Dict[str, float], Dict, Dict]:
    detector.eval()
    total_primary_loss = 0.0
    total_secondary_loss = 0.0
    total_conflict_loss = 0.0
    total_samples = 0

    all_primary_preds: List[int] = []
    all_primary_labels: List[int] = []
    all_secondary_preds: List[int] = []
    all_secondary_labels: List[int] = []
    all_secondary_mask: List[int] = []

    for batch in dataloader:
        loss, metrics = compute_loss(
            detector, batch, primary_criterion, secondary_criterion, device,
            secondary_weight, use_soft_labels, conflict_score_weight
        )

        batch_size = len(batch["primary"])
        total_primary_loss += metrics["primary_loss"] * batch_size
        total_secondary_loss += metrics["secondary_loss"] * batch_size
        total_conflict_loss += metrics["conflict_score_loss"] * batch_size
        total_samples += batch_size

        # Collect predictions
        for hidden_states, p_label, s_label in zip(
            batch["hidden_states"], batch["primary"], batch["secondary"]
        ):
            hs = {k: v.to(device).unsqueeze(0) for k, v in hidden_states.items()}
            primary_logits, secondary_logits, _, _ = detector(hs, prev_state=None)

            primary_pred = primary_logits.argmax(dim=-1).item()
            all_primary_preds.append(primary_pred)
            all_primary_labels.append(PRIMARY_IDX_MAP[p_label])

            if s_label is not None:
                secondary_pred = secondary_logits.argmax(dim=-1).item()
                all_secondary_preds.append(secondary_pred)
                all_secondary_labels.append(SECONDARY_IDX_MAP[s_label])
                all_secondary_mask.append(1)

    avg_metrics = {
        "primary_loss": total_primary_loss / total_samples,
        "secondary_loss": total_secondary_loss / total_samples,
        "conflict_score_loss": total_conflict_loss / total_samples,
        "total_loss": (total_primary_loss + total_secondary_loss + total_conflict_loss) / total_samples,
    }

    # Primary classification report
    present_primary_labels = sorted(set(all_primary_labels))
    primary_target_names = [PredictiveCodingDetector.PRIMARY_LABELS[i] for i in present_primary_labels]
    primary_report = classification_report(
        all_primary_labels,
        all_primary_preds,
        labels=present_primary_labels,
        target_names=primary_target_names,
        output_dict=True,
        zero_division=0,
    )
    primary_macro_f1 = f1_score(
        all_primary_labels, all_primary_preds,
        labels=present_primary_labels,
        average="macro", zero_division=0.0
    )

    # Secondary classification report
    secondary_report = None
    secondary_macro_f1 = 0.0
    if sum(all_secondary_mask) > 0:
        present_secondary_labels = sorted(set(all_secondary_labels))
        secondary_target_names = [PredictiveCodingDetector.SECONDARY_LABELS[i] for i in present_secondary_labels]
        secondary_report = classification_report(
            all_secondary_labels,
            all_secondary_preds,
            labels=present_secondary_labels,
            target_names=secondary_target_names,
            output_dict=True,
            zero_division=0,
        )
        secondary_macro_f1 = f1_score(
            all_secondary_labels, all_secondary_preds,
            labels=present_secondary_labels,
            average="macro", zero_division=0.0
        )

    return avg_metrics, {
        "primary": primary_report,
        "secondary": secondary_report,
        "primary_macro_f1": primary_macro_f1,
        "secondary_macro_f1": secondary_macro_f1,
    }, avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the Predictive Coding conflict detector")
    parser.add_argument("--data", type=str, default=str(_PROJECT_ROOT / "data" / "acc_training" / "generated_conflict_data.jsonl"))
    parser.add_argument("--save_dir", type=str, default=str(_PROJECT_ROOT / "adapters" / "acc_conflict_detector"))
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layer_pairs", type=str, default=None, help='e.g. "-4,-2,-2,-1"')
    parser.add_argument("--temporal_decay", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="pc-conflict-detector")

    # Loss options
    parser.add_argument("--secondary_weight", type=float, default=0.5, help="Weight for secondary head loss")
    parser.add_argument("--use_soft_labels", action="store_true", help="Use soft targets if available in data")
    parser.add_argument("--conflict_score_weight", type=float, default=0.0, help="Weight for auxiliary conflict score MSE loss")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for hard labels (0 = disabled)")
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma (0 = disabled, use standard CE)")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping max norm")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # WandB
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: v for k, v in vars(args).items() if k not in ("wandb_project", "wandb_run_name")},
        )

    # Load data
    print(f"Loading data from {args.data} ...")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    records = load_records(args.data)
    print(f"Loaded {len(records)} token records.")
    if len(records) == 0:
        raise ValueError("No records found in data file.")

    # Validate labels
    valid_old_labels = set(OLD_TO_PRIMARY.keys())
    bad_labels = [r["label"] for r in records if r["label"] not in valid_old_labels]
    if bad_labels:
        raise ValueError(f"Unknown labels found: {set(bad_labels)}")

    # Class balance report
    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15s}: {count:5d} ({100*count/len(records):.1f}%)")

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

    # Auto-detect dimensions
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        sample = records[0]
        if "hidden_states" in sample:
            first_layer = list(sample["hidden_states"].keys())[0]
            hidden_dim = len(sample["hidden_states"][first_layer])
        elif "hidden_state" in sample:
            hidden_dim = len(sample["hidden_state"])
        else:
            raise ValueError("Cannot auto-detect hidden_dim from data")
        print(f"Auto-detected hidden_dim: {hidden_dim}")

    # Parse layer pairs
    if args.layer_pairs:
        parts = [int(x) for x in args.layer_pairs.split(",")]
        layer_pairs = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    else:
        if "hidden_states" in records[0]:
            available_layers = sorted([int(k) for k in records[0]["hidden_states"].keys()])
            layer_pairs = []
            for i in range(len(available_layers) - 1):
                layer_pairs.append((available_layers[i], available_layers[i + 1]))
            if not layer_pairs:
                layer_pairs = [(-1, -1)]
        else:
            layer_pairs = [(-4, -1)]
    print(f"Layer pairs: {layer_pairs}")

    # Validate: detector layer_pairs must be compatible with data layers
    data_layers = set()
    for r in records[:10]:  # sample first 10
        if "hidden_states" in r:
            data_layers.update(int(k) for k in r["hidden_states"].keys())
    required_layers = set(idx for pair in layer_pairs for idx in pair)
    missing = required_layers - data_layers
    if missing:
        print(f"WARNING: Layer pairs require layers {sorted(required_layers)}, "
              f"but data only has {sorted(data_layers)}. Missing: {sorted(missing)}")

    # DataLoaders
    train_ds = MultiLayerTokenDataset(train_records, use_soft_labels=args.use_soft_labels)
    val_ds = MultiLayerTokenDataset(val_records, use_soft_labels=args.use_soft_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    # Model
    detector = PredictiveCodingDetector(
        hidden_dim=hidden_dim,
        layer_pairs=layer_pairs,
        temporal_decay=args.temporal_decay,
        dropout=args.dropout,
    )
    detector = detector.to(device)
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"PredictiveCodingDetector: {n_params/1e6:.2f}M parameters")
    print(f"  Primary labels: {detector.PRIMARY_LABELS}")
    print(f"  Secondary labels: {detector.SECONDARY_LABELS}")
    print(f"  Temporal decay: {detector.temporal_decay}")
    print(f"  Soft labels: {args.use_soft_labels}")
    print(f"  Focal gamma: {args.focal_gamma}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Conflict score weight: {args.conflict_score_weight}")
    print(f"  Gradient clip: {args.grad_clip}")

    # Optimizer
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss functions
    if args.focal_gamma > 0:
        primary_criterion = FocalLoss(gamma=args.focal_gamma)
        secondary_criterion = FocalLoss(gamma=args.focal_gamma, reduction="none")
    else:
        primary_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        secondary_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Training loop with dual early stopping
    best_primary_f1 = -1.0
    best_secondary_f1 = -1.0
    epochs_without_improvement = 0

    history = {
        "train_primary_loss": [],
        "train_secondary_loss": [],
        "val_primary_loss": [],
        "val_secondary_loss": [],
        "primary_f1": [],
        "secondary_f1": [],
    }

    print(f"\n{'Epoch':>6s} {'TrainLoss':>10s} {'ValLoss':>9s} {'PrimF1':>8s} {'SecF1':>8s}")
    print("-" * 50)

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            detector, train_loader, optimizer, primary_criterion, secondary_criterion,
            device, args.secondary_weight, args.use_soft_labels, args.conflict_score_weight, args.grad_clip
        )
        val_metrics, reports, _ = evaluate(
            detector, val_loader, primary_criterion, secondary_criterion,
            device, args.secondary_weight, args.use_soft_labels, args.conflict_score_weight
        )

        primary_f1 = reports["primary_macro_f1"]
        secondary_f1 = reports["secondary_macro_f1"]
        scheduler.step(primary_f1)

        # Record history
        history["train_primary_loss"].append(train_metrics["primary_loss"])
        history["train_secondary_loss"].append(train_metrics["secondary_loss"])
        history["val_primary_loss"].append(val_metrics["primary_loss"])
        history["val_secondary_loss"].append(val_metrics["secondary_loss"])
        history["primary_f1"].append(primary_f1)
        history["secondary_f1"].append(secondary_f1)

        print(
            f"{epoch:6d} {train_metrics['total_loss']:10.4f} "
            f"{val_metrics['total_loss']:9.4f} {primary_f1:8.4f} {secondary_f1:8.4f}"
        )

        # Per-class metrics
        for label_type, report in [("primary", reports["primary"]), ("secondary", reports.get("secondary"))]:
            if report is None:
                continue
            for name in list(report.keys()):
                if name in ("accuracy", "macro avg", "weighted avg"):
                    continue
                p = report[name]["precision"]
                r = report[name]["recall"]
                f1 = report[name]["f1-score"]
                support = int(report[name]["support"])
                print(f"  [{label_type:9s}] {name:12s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={support}")

        # WandB logging
        if args.wandb_project:
            import wandb
            log_dict = {
                "epoch": epoch,
                "train_loss": train_metrics["total_loss"],
                "train_primary_loss": train_metrics["primary_loss"],
                "train_secondary_loss": train_metrics["secondary_loss"],
                "val_loss": val_metrics["total_loss"],
                "val_primary_loss": val_metrics["primary_loss"],
                "val_secondary_loss": val_metrics["secondary_loss"],
                "primary_macro_f1": primary_f1,
                "secondary_macro_f1": secondary_f1,
            }
            for label_type, report in [("primary", reports["primary"]), ("secondary", reports.get("secondary"))]:
                if report:
                    for name in list(report.keys()):
                        if name not in ("accuracy", "macro avg", "weighted avg"):
                            log_dict[f"{label_type}_{name}_f1"] = report[name]["f1-score"]
            wandb.log(log_dict)

        # Checkpointing: best primary
        improved = False
        if primary_f1 > best_primary_f1:
            best_primary_f1 = primary_f1
            improved = True
            save_checkpoint(
                detector, Path(args.save_dir),
                metrics={"epoch": epoch, "primary_macro_f1": primary_f1, "secondary_macro_f1": secondary_f1},
                suffix="_best_primary"
            )
            print(f"  [*] New best PRIMARY model (macro_f1={primary_f1:.4f})")

        # Checkpointing: best secondary
        if secondary_f1 > best_secondary_f1:
            best_secondary_f1 = secondary_f1
            improved = True
            save_checkpoint(
                detector, Path(args.save_dir),
                metrics={"epoch": epoch, "primary_macro_f1": primary_f1, "secondary_macro_f1": secondary_f1},
                suffix="_best_secondary"
            )
            print(f"  [*] New best SECONDARY model (macro_f1={secondary_f1:.4f})")

        # Early stopping
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # Save final history
    history_path = Path(args.save_dir) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Finalize
    if args.wandb_project:
        import wandb
        wandb.summary["best_primary_macro_f1"] = best_primary_f1
        wandb.summary["best_secondary_macro_f1"] = best_secondary_f1
        wandb.finish()

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best primary macro-F1:   {best_primary_f1:.4f}")
    print(f"  Best secondary macro-F1: {best_secondary_f1:.4f}")
    print(f"  Checkpoints saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
