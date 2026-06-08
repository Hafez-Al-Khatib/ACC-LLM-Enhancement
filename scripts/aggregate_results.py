#!/usr/bin/env python3
"""
Aggregate and visualize ACC LLM experimental results.

Usage:
    python scripts/aggregate_results.py --input results/ --output results/summary/

This script:
1. Loads all JSON result files from a directory
2. Computes summary statistics across conditions and datasets
3. Performs pairwise statistical tests (Wilcoxon signed-rank)
4. Generates LaTeX tables and matplotlib plots for the paper
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests disabled.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


def load_results(input_dir):
    """Load all JSON result files from directory."""
    results = []
    pattern = os.path.join(input_dir, "*.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = os.path.basename(path)
                results.append(data)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    print(f"Loaded {len(results)} result files from {input_dir}")
    return results


def extract_metrics(results):
    """Extract structured metrics from raw result dicts."""
    rows = []
    for r in results:
        condition = r.get("condition", r.get("model_name", "unknown"))
        dataset = r.get("dataset", r.get("vertical", "unknown"))
        metrics = r.get("metrics", r)
        
        row = {
            "condition": condition,
            "dataset": dataset,
            "source": r.get("_source_file", ""),
        }
        
        # Primary metrics
        for key in ["hallucination_f1", "contradiction_rate", "calibration_error", 
                    "perplexity", "latency_ms", "memory_gb"]:
            if key in metrics:
                row[key] = metrics[key]
        
        rows.append(row)
    return rows


def summarize_by_condition(rows):
    """Group by condition and compute mean/std."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    
    summary = {}
    numeric_keys = [k for k in rows[0].keys() if k not in ("condition", "dataset", "source")]
    
    for condition, group in sorted(groups.items()):
        summary[condition] = {}
        for key in numeric_keys:
            values = [r[key] for r in group if key in r and isinstance(r[key], (int, float))]
            if values:
                summary[condition][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
    return summary


def pairwise_tests(rows, baseline="Base", metric="hallucination_f1"):
    """Wilcoxon signed-rank test between each condition and baseline."""
    if not HAS_SCIPY:
        return {}
    
    # Group by (dataset, condition)
    by_dataset_condition = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ds = row["dataset"]
        cond = row["condition"]
        if metric in row and isinstance(row[metric], (int, float)):
            by_dataset_condition[ds][cond].append(row[metric])
    
    results = {}
    conditions = set()
    for ds in by_dataset_condition:
        conditions.update(by_dataset_condition[ds].keys())
    
    for cond in sorted(conditions):
        if cond == baseline:
            continue
        pvalues = []
        for ds in by_dataset_condition:
            base_vals = by_dataset_condition[ds].get(baseline, [])
            cond_vals = by_dataset_condition[ds].get(cond, [])
            if len(base_vals) >= 3 and len(cond_vals) >= 3:
                # Paired test requires same number of samples; use unpaired if unequal
                if len(base_vals) == len(cond_vals):
                    stat, p = stats.wilcoxon(base_vals, cond_vals)
                else:
                    stat, p = stats.mannwhitneyu(base_vals, cond_vals, alternative="two-sided")
                pvalues.append(p)
        
        if pvalues:
            results[cond] = {
                "median_p": float(np.median(pvalues)),
                "min_p": float(np.min(pvalues)),
                "significant_at_05": sum(1 for p in pvalues if p < 0.05),
                "total_comparisons": len(pvalues),
            }
    return results


def generate_latex_table(summary, metric="hallucination_f1", caption="Results"):
    """Generate a LaTeX table from summary statistics."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Condition & Mean & Std & N \\\\")
    lines.append("\\midrule")
    
    for condition, metrics in sorted(summary.items()):
        if metric in metrics:
            m = metrics[metric]
            lines.append(
                f"{condition} & {m['mean']:.3f} & {m['std']:.3f} & {m['n']} \\\\"
            )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_plots(rows, output_dir):
    """Generate bar plots and box plots for key metrics."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar plot: mean hallucination F1 by condition
    summary = summarize_by_condition(rows)
    conditions = sorted(summary.keys())
    metric = "hallucination_f1"
    
    means = []
    stds = []
    labels = []
    for cond in conditions:
        if metric in summary[cond]:
            means.append(summary[cond][metric]["mean"])
            stds.append(summary[cond][metric]["std"])
            labels.append(cond)
    
    if means:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Hallucination F1")
        ax.set_title("Mean Hallucination F1 by Condition")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        path = os.path.join(output_dir, "hallucination_f1_by_condition.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot: {path}")
    
    # Box plot: metric distribution by condition
    conditions_with_data = []
    data_by_condition = []
    for cond in conditions:
        values = [r[metric] for r in rows if r["condition"] == cond and metric in r]
        if values:
            conditions_with_data.append(cond)
            data_by_condition.append(values)
    
    if data_by_condition:
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data_by_condition, labels=conditions_with_data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax.set_ylabel("Hallucination F1")
        ax.set_title("Distribution of Hallucination F1 by Condition")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(output_dir, "hallucination_f1_distribution.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate ACC LLM results")
    parser.add_argument("--input", default="results/", help="Directory with JSON result files")
    parser.add_argument("--output", default="results/summary/", help="Output directory")
    parser.add_argument("--baseline", default="Base", help="Baseline condition name")
    args = parser.parse_args()
    
    results = load_results(args.input)
    if not results:
        print("No results found. Exiting.")
        sys.exit(1)
    
    rows = extract_metrics(results)
    summary = summarize_by_condition(rows)
    tests = pairwise_tests(rows, baseline=args.baseline)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Save summary JSON
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "statistical_tests": tests,
            "total_samples": len(rows),
        }, f, indent=2)
    print(f"Saved summary: {summary_path}")
    
    # Save LaTeX table
    latex = generate_latex_table(summary, caption="ACC LLM Results by Condition")
    latex_path = os.path.join(args.output, "results_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX table: {latex_path}")
    
    # Generate plots
    generate_plots(rows, args.output)
    
    # Print human-readable summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for condition, metrics in sorted(summary.items()):
        print(f"\n{condition}:")
        for key, vals in sorted(metrics.items()):
            print(f"  {key}: {vals['mean']:.4f} ± {vals['std']:.4f} (n={vals['n']})")
    
    if tests:
        print("\n" + "="*60)
        print("STATISTICAL TESTS vs BASELINE")
        print("="*60)
        for cond, res in sorted(tests.items()):
            sig = "YES" if res["significant_at_05"] > 0 else "NO"
            print(f"\n{cond}:")
            print(f"  Median p-value: {res['median_p']:.4f}")
            print(f"  Significant comparisons: {res['significant_at_05']}/{res['total_comparisons']} ({sig})")


if __name__ == "__main__":
    main()
