#!/usr/bin/env python3
"""Robust HuggingFace model downloader with retries and resume support.

Handles flaky connections by retrying individual shards with exponential backoff.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError


def download_with_retries(repo_id: str, local_dir: Path, max_retries: int = 10, delay: int = 5):
    """Download all files from a HuggingFace repo with per-file retries."""
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files for {repo_id}...")
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Failed to list files: {e}")
        sys.exit(1)

    print(f"Found {len(files)} files. Starting download...")

    for filepath in files:
        dest = local_dir / filepath
        if dest.exists():
            print(f"  [exists] {filepath}")
            continue

        attempt = 0
        while attempt < max_retries:
            try:
                print(f"  [downloading] {filepath} (attempt {attempt + 1}/{max_retries})")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filepath,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"  [done] {filepath}")
                break
            except Exception as e:
                attempt += 1
                print(f"  [error] {e}")
                if attempt < max_retries:
                    wait = delay * (2 ** attempt)
                    print(f"  [retry] waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [failed] {filepath} after {max_retries} attempts")
                    raise

    print(f"\nAll files downloaded to: {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", default="Qwen/Qwen2.5-7B", help="HuggingFace model repo ID")
    parser.add_argument("--local-dir", type=Path, help="Destination directory")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--delay", type=int, default=5, help="Base retry delay in seconds")
    parser.add_argument("--use-snapshot", action="store_true", help="Use snapshot_download instead of per-file")
    args = parser.parse_args()

    if args.local_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        model_name_safe = args.repo_id.replace("/", "_")
        args.local_dir = repo_root / "models" / model_name_safe

    print(f"Downloading {args.repo_id} -> {args.local_dir}")

    if args.use_snapshot:
        # Standard snapshot download (less robust but simpler)
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(args.local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    else:
        download_with_retries(args.repo_id, args.local_dir, args.max_retries, args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
