"""Entrypoint for Phase 2 Step 1 and Step 2.

Step 1: Build a labelled subset (minimum 2000 reviews) from the dataset.
Step 2: Run data exploration on that subset and save report artifacts.

Run with:
    python -m src.phase2.run_phase2_steps12 --data_path data/AMAZON_FASHION.json --subset_size 2000 --seed 42
"""

import argparse
import json
import os
import sys

from src.io_utils import load_jsonl
from src.phase2.preprocess_phase2 import prepare_phase2_dataframe, select_stratified_subset
from src.phase2.explore_phase2 import run_step2_exploration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 (Step 1 + Step 2) - Subset + Exploration")
    parser.add_argument(
        "--data_path",
        default="data/AMAZON_FASHION.json",
        help="Path to dataset (.json/.jsonl/.json.gz/.jsonl.gz).",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=2000,
        help="Target subset size for Phase 2 Step 1 (minimum recommended: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/phase 2",
        help="Root output directory; step subfolders are created inside it.",
    )
    return parser.parse_args(argv)


def resolve_data_path(path: str) -> str | None:
    if os.path.exists(path):
        return path

    parent = os.path.dirname(path) or "data"
    candidates = [
        "AMAZON_FASHION.json",
        "AMAZON_FASHION.jsonl",
        "AMAZON_FASHION.json.gz",
        "AMAZON_FASHION.jsonl.gz",
        "AMAZON_FASHION_5.json",
        "AMAZON_FASHION_5.json.gz",
    ]
    for name in candidates:
        candidate_path = os.path.join(parent, name)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 64)
    print("  PHASE 2 - STEP 1 + STEP 2")
    print("=" * 64)
    print(f"  data_path   : {args.data_path}")
    print(f"  subset_size : {args.subset_size}")
    print(f"  seed        : {args.seed}")
    print(f"  out_dir     : {args.out_dir}")
    print("=" * 64 + "\n")

    resolved = resolve_data_path(args.data_path)
    if resolved is None:
        print(f"[main] ERROR: Could not find dataset path from '{args.data_path}'.", file=sys.stderr)
        sys.exit(1)
    if resolved != args.data_path:
        print(f"[main] '{args.data_path}' not found; using '{resolved}' instead.")

    if args.subset_size < 2000:
        print("[main] NOTE: subset_size is below 2000. Phase 2 requirement is minimum 2000 reviews.")

    raw_df = load_jsonl(resolved)

    print("\n[main] Step 1 - Preparing labelled/clean dataframe ...")
    prepared_df = prepare_phase2_dataframe(raw_df)
    subset_df = select_stratified_subset(prepared_df, n=args.subset_size, seed=args.seed)

    phase2_root = args.out_dir
    step1_dir = os.path.join(phase2_root, "step 1")
    step2_dir = os.path.join(phase2_root, "step 2")
    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(step2_dir, exist_ok=True)

    step1_subset_path = os.path.join(step1_dir, "step1_subset_labeled.csv")
    subset_df.to_csv(step1_subset_path, index=False)

    step1_summary = {
        "raw_rows_loaded": int(len(raw_df)),
        "rows_after_cleaning_labelling": int(len(prepared_df)),
        "subset_rows_saved": int(len(subset_df)),
        "subset_sentiment_distribution": {k: int(v) for k, v in subset_df["sentiment"].value_counts().to_dict().items()},
        "subset_rating_distribution": {str(k): int(v) for k, v in subset_df["overall"].value_counts().sort_index().to_dict().items()},
    }
    step1_summary_path = os.path.join(step1_dir, "step1_summary.json")
    with open(step1_summary_path, "w", encoding="utf-8") as fh:
        json.dump(step1_summary, fh, indent=2)

    print("\n[main] Step 2 - Running exploration on Step 1 subset ...")
    step2_summary = run_step2_exploration(subset_df, out_dir=step2_dir)
    print("[main] Step 2 summary (key metrics):")
    print(json.dumps(step2_summary, indent=2))

    print("\n" + "=" * 64)
    print("  DONE. Output files:")
    print(f"    {step1_subset_path}")
    print(f"    {step1_summary_path}")
    print(f"    {os.path.join(step2_dir, 'step2_summary.json')}")
    print(f"    {os.path.join(step2_dir, 'rating_distribution.csv')}")
    print(f"    {os.path.join(step2_dir, 'sentiment_distribution.csv')}")
    print(f"    {os.path.join(step2_dir, 'figures')}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
