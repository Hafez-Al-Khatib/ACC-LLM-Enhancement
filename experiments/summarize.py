"""Summarize ablation results from experiments/results/*.jsonl."""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(results_file: str) -> pd.DataFrame:
    records = []
    with open(results_file, "r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    total = len(df)
    passed = df["success"].sum()
    print(f"Runs: {passed}/{total} successful")

    if "elapsed_seconds" in df.columns:
        print(f"Total time: {df['elapsed_seconds'].sum() / 60:.1f} minutes")
        print(f"Mean run time: {df['elapsed_seconds'].mean() / 60:.1f} minutes")

    # Try to extract the ablated parameter from config path
    # Path pattern: .../vertical_dataset_ablateValue_hardware_ts.yaml
    if "config" in df.columns:
        df["ablated_value"] = df["config"].str.extract(r"_([a-z]+)(\d+)_.*\.yaml")

    print("\nPer-run results:")
    for _, row in df.iterrows():
        status = "✓" if row["success"] else "✗"
        print(f"  {status} {row.get('config', 'unknown')} | {row.get('elapsed_seconds', 0):.0f}s")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to .jsonl results file")
    args = parser.parse_args()

    df = load_results(args.results)
    print_summary(df)
