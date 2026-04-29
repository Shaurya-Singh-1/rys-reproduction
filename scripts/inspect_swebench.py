#!/usr/bin/env python3
"""Interactively inspect SWE-bench samples for localization dataset curation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import resolve_dataset_name


def import_load_dataset():
    """Import Hugging Face datasets without being shadowed by ./datasets."""
    original_path = list(sys.path)
    root_resolved = ROOT.resolve()
    sys.path = [
        item
        for item in sys.path
        if Path(item or os.getcwd()).resolve() != root_resolved
    ]
    cached = sys.modules.get("datasets")
    if cached is not None and not hasattr(cached, "load_dataset"):
        sys.modules.pop("datasets", None)
    try:
        return importlib.import_module("datasets").load_dataset
    finally:
        sys.path = original_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SWE-bench samples one at a time.")
    parser.add_argument("--subset", default="lite", help="Dataset alias or full dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--start", type=int, default=0, help="First dataset index to show.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to show.")
    parser.add_argument("--patch-chars", type=int, default=1200, help="Number of patch chars to print.")
    parser.add_argument("--problem-chars", type=int, default=1600, help="Number of problem chars to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dataset = import_load_dataset()
    dataset_name = resolve_dataset_name(args.subset)
    dataset = load_dataset(dataset_name, split=args.split)
    stop = len(dataset) if args.limit is None else min(len(dataset), args.start + args.limit)

    for idx in range(args.start, stop):
        sample = dataset[idx]
        print("=" * 80)
        print("Index:", idx)
        print("Instance ID:", sample.get("instance_id", ""))
        print("Repo:", sample.get("repo", ""))
        print("Base commit:", sample.get("base_commit", ""))
        print("\nProblem:")
        print(str(sample.get("problem_statement", ""))[: args.problem_chars])
        print("\nPatch:")
        print(str(sample.get("patch", ""))[: args.patch_chars])

        try:
            input("\nPress Enter for next sample, or Ctrl+C to stop...")
        except KeyboardInterrupt:
            print()
            break


if __name__ == "__main__":
    main()
