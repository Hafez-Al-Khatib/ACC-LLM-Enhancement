#!/usr/bin/env python3
"""Visualize ablation study results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ablation_study_1.5b.json")
    parser.add_argument("--output", default="results/figures/ablation_accuracy.png")
    args = parser.parse_args()

    with open(_PROJECT_ROOT / args.input) as f:
        data = json.load(f)

    summary = data["summary"]
    names = list(summary.keys())
    accs = [summary[n]["accuracy"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if "Baseline" in n else "#3498db" for n in names]
    bars = ax.barh(names, accs, color=colors, edgecolor="black", linewidth=0.5)

    # Highlight logit-shift bars
    for bar, name in zip(bars, names):
        if "logit-shift" in name:
            bar.set_color("#2ecc71")
        elif "phrase" in name:
            bar.set_color("#f39c12")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study: Intervention Strategies", fontsize=14, fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="random baseline")

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height() / 2, f"{acc:.1f}%", va="center", fontsize=9)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="No intervention"),
        Patch(facecolor="#f39c12", label="Phrase intervention"),
        Patch(facecolor="#2ecc71", label="Logit-shift intervention"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
