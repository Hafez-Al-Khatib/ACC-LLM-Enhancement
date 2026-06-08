"""Diagnostic visualization tool for PredictiveCodingDetector.

Generates plots and reports to help understand:
  1. Prediction errors across layer pairs
  2. Leaky integrator state trajectories over time
  3. Layer contribution attributions per classification
  4. Conflict score evolution during generation

Usage::

    python scripts/visualize_detector.py \
        --records data/acc_training/test_pc_data.jsonl \
        --detector adapters/test_pc_detector \
        --output results/visualizations/

Or with synthetic data for quick testing::

    python scripts/visualize_detector.py --synthetic --output results/visualizations/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.acc_conflict_detector import PredictiveCodingDetector

# Matplotlib imports (optional — graceful degradation)
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Visualizations will be skipped.")


# ---------------------------------------------------------------------------
# Data loading
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


def load_detector_checkpoint(checkpoint_dir: str) -> PredictiveCodingDetector:
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "config.json"
    weights_path = checkpoint_path / "detector.pt"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    detector = PredictiveCodingDetector(
        hidden_dim=config["hidden_dim"],
        layer_pairs=[tuple(p) for p in config["layer_pairs"]],
        temporal_decay=config.get("temporal_decay", 0.7),
    )
    detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector.eval()
    return detector


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_sequence(
    detector: PredictiveCodingDetector,
    num_tokens: int = 20,
    seed: int = 42,
) -> List[Dict[int, torch.Tensor]]:
    """Generate a synthetic hidden-state sequence with injected anomalies."""
    rng = np.random.default_rng(seed)
    sequence = []

    # Collect all unique layer indices from the detector's layer pairs
    layer_indices = sorted(set(idx for pair in detector.layer_pairs for idx in pair))
    hidden_dim = detector.hidden_dim

    for t in range(num_tokens):
        hs = {}
        for idx in layer_indices:
            # Base hidden state
            base = rng.standard_normal(hidden_dim).astype(np.float32)

            # Inject anomaly at token 10-12 (simulate hallucination)
            if 10 <= t <= 12:
                # Add large noise to later layers to create prediction errors
                if idx >= max(layer_indices) - 1:
                    base += rng.standard_normal(hidden_dim).astype(np.float32) * 3.0

            hs[idx] = torch.from_numpy(base)
        sequence.append(hs)

    return sequence


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def plot_prediction_errors(
    detector: PredictiveCodingDetector,
    sequence: List[Dict[int, torch.Tensor]],
    output_path: Path,
):
    """Plot prediction errors for each layer pair across time."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    errors = []
    for hs in sequence:
        pe = detector.get_prediction_errors(hs)
        errors.append([pe[pair] for pair in detector.layer_pairs])
    errors = np.array(errors)  # (time, num_pairs)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, pair in enumerate(detector.layer_pairs):
        ax.plot(errors[:, i], label=f"Layer {pair[0]} → {pair[1]}", marker="o", markersize=3)

    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Prediction Error (MSE)")
    ax.set_title("Hierarchical Prediction Errors Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_state_trajectory(
    detector: PredictiveCodingDetector,
    sequence: List[Dict[int, torch.Tensor]],
    output_path: Path,
):
    """Plot leaky integrator state trajectory over time."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    states = []
    state = None
    for hs in sequence:
        _, _, _, state = detector(hs, prev_state=state)
        states.append(state.squeeze(0).detach().cpu().numpy())
    states = np.array(states)  # (time, num_pairs)

    fig, axes = plt.subplots(
        nrows=detector.num_pairs,
        ncols=1,
        figsize=(10, 2 * detector.num_pairs),
        sharex=True,
    )
    if detector.num_pairs == 1:
        axes = [axes]

    for i, pair in enumerate(detector.layer_pairs):
        ax = axes[i]
        ax.plot(states[:, i], color="C0", marker="o", markersize=3)
        ax.set_ylabel(f"State {pair}")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.2)

    axes[-1].set_xlabel("Generation Step")
    fig.suptitle("Leaky Integrator State Trajectories", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_contributions(
    detector: PredictiveCodingDetector,
    sequence: List[Dict[int, torch.Tensor]],
    output_path: Path,
):
    """Plot layer contribution heatmap over time."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    contributions = []
    for hs in sequence:
        contrib = detector.get_layer_contributions(hs, target="conflict_score")
        contributions.append([contrib[pair] for pair in detector.layer_pairs])
    contributions = np.array(contributions)  # (time, num_pairs)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        contributions.T,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_yticks(range(detector.num_pairs))
    ax.set_yticklabels([f"{p[0]} → {p[1]}" for p in detector.layer_pairs])
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Layer Pair")
    ax.set_title("Layer Contribution to Conflict Score (Gradient Attribution)")
    plt.colorbar(im, ax=ax, label="Attribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_conflict_score_trajectory(
    detector: PredictiveCodingDetector,
    sequence: List[Dict[int, torch.Tensor]],
    output_path: Path,
):
    """Plot conflict score and primary label probabilities over time."""
    if not _MATPLOTLIB_AVAILABLE:
        return

    scores = []
    primary_probs = {label: [] for label in detector.PRIMARY_LABELS}

    state = None
    for hs in sequence:
        result = detector.classify(hs, prev_state=state)
        state = result["next_state"]
        scores.append(result["conflict_score"])
        for label in detector.PRIMARY_LABELS:
            primary_probs[label].append(result["primary_probs"][label])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

    # Top: conflict score
    ax = axes[0]
    ax.plot(scores, color="red", linewidth=2, label="Conflict Score")
    ax.set_ylabel("Conflict Score")
    ax.set_ylim(0, 1)
    ax.set_title("Conflict Score Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Bottom: primary label probabilities
    ax = axes[1]
    for label, probs in primary_probs.items():
        ax.plot(probs, label=label, marker="o", markersize=3)
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Primary Label Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_text_report(
    detector: PredictiveCodingDetector,
    sequence: List[Dict[int, torch.Tensor]],
    output_path: Path,
):
    """Generate a text report with per-token explanations."""
    lines = []
    lines.append("=" * 70)
    lines.append("PredictiveCodingDetector Diagnostic Report")
    lines.append("=" * 70)
    lines.append(f"Layer pairs: {detector.layer_pairs}")
    lines.append(f"Temporal decay (alpha): {detector.temporal_decay}")
    lines.append(f"Hidden dim: {detector.hidden_dim}")
    lines.append(f"Num parameters: {sum(p.numel() for p in detector.parameters()) / 1e6:.2f}M")
    lines.append("")

    state = None
    for t, hs in enumerate(sequence):
        explanation = detector.explain(hs, prev_state=state)
        state = explanation["classification"]["next_state"]

        lines.append(f"--- Step {t} ---")
        lines.append(f"  Primary: {explanation['classification']['primary']} "
                    f"(p={explanation['classification']['primary_probs'][explanation['classification']['primary']]:.3f})")
        if explanation["classification"]["secondary"]:
            lines.append(f"  Secondary: {explanation['classification']['secondary']} "
                        f"(p={explanation['classification']['secondary_probs'][explanation['classification']['secondary']]:.3f})")
        lines.append(f"  Conflict score: {explanation['classification']['conflict_score']:.4f}")
        lines.append("  Prediction errors:")
        for pair, err in explanation["prediction_errors"].items():
            lines.append(f"    {pair}: {err:.6f}")
        lines.append("  Layer contributions (conflict_score):")
        for pair, contrib in explanation["layer_contributions"]["conflict_score"].items():
            lines.append(f"    {pair}: {contrib:.6f}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize PredictiveCodingDetector diagnostics")
    parser.add_argument("--records", type=str, default=None, help="Path to JSONL records")
    parser.add_argument("--detector", type=str, default=None, help="Path to detector checkpoint dir")
    parser.add_argument("--output", type=str, default=str(_PROJECT_ROOT / "results" / "visualizations"))
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or create detector
    # ------------------------------------------------------------------
    if args.detector:
        detector = load_detector_checkpoint(args.detector)
        print(f"Loaded detector from {args.detector}")
    else:
        # Create an untrained detector for demo
        detector = PredictiveCodingDetector(
            hidden_dim=args.hidden_dim,
            layer_pairs=[(-4, -2), (-2, -1)],
            temporal_decay=0.7,
            dropout=0.0,
        )
        print("Using untrained detector (synthetic/demo mode)")

    # ------------------------------------------------------------------
    # Load or create sequence
    # ------------------------------------------------------------------
    if args.synthetic or not args.records:
        sequence = generate_synthetic_sequence(
            detector=detector,
            num_tokens=args.num_tokens,
            seed=args.seed,
        )
        print(f"Generated synthetic sequence: {len(sequence)} tokens")
    else:
        records = load_records(args.records)
        # Convert records to tensor dicts
        sequence = []
        for rec in records[:args.num_tokens]:
            hs = {}
            for layer_idx, vec in rec.get("hidden_states", {}).items():
                hs[int(layer_idx)] = torch.tensor(vec, dtype=torch.float32)
            sequence.append(hs)
        print(f"Loaded {len(sequence)} tokens from {args.records}")

    if len(sequence) == 0:
        print("ERROR: No data to visualize. Use --synthetic flag.")
        return

    # ------------------------------------------------------------------
    # Generate visualizations
    # ------------------------------------------------------------------
    print("\nGenerating visualizations...")

    plot_prediction_errors(
        detector, sequence, output_dir / "prediction_errors.png"
    )
    plot_state_trajectory(
        detector, sequence, output_dir / "state_trajectories.png"
    )
    plot_layer_contributions(
        detector, sequence, output_dir / "layer_contributions.png"
    )
    plot_conflict_score_trajectory(
        detector, sequence, output_dir / "conflict_trajectory.png"
    )
    generate_text_report(
        detector, sequence, output_dir / "diagnostic_report.txt"
    )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
