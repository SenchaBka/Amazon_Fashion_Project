"""Entrypoint for Phase 2 Step 3.

Step 3: Preprocess text and build TF-IDF representation.

Run with:
    python -m src.phase2.run_phase2_step3
"""

import argparse
import json
import os
import sys

from src.phase2.represent_phase2 import run_step3_tfidf


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Step 3 - Preprocessing + TF-IDF")
    parser.add_argument(
        "--subset_csv",
        default="outputs/phase 2/step 1/step1_subset_labeled.csv",
        help="Path to Step 1 subset CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/phase 2/step 3",
        help="Output directory for Step 3 artifacts.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=20000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--max_df",
        type=float,
        default=0.95,
        help="Maximum document frequency proportion for TF-IDF terms.",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Maximum n-gram size (minimum is always 1).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 64)
    print("  PHASE 2 - STEP 3")
    print("=" * 64)
    print(f"  subset_csv   : {args.subset_csv}")
    print(f"  out_dir      : {args.out_dir}")
    print(f"  max_features : {args.max_features}")
    print(f"  min_df       : {args.min_df}")
    print(f"  max_df       : {args.max_df}")
    print(f"  ngram_range  : (1, {args.ngram_max})")
    print("=" * 64 + "\n")

    if not os.path.exists(args.subset_csv):
        print(f"[main] ERROR: Step 1 subset not found at '{args.subset_csv}'.", file=sys.stderr)
        sys.exit(1)

    if args.ngram_max < 1:
        print("[main] ERROR: --ngram_max must be >= 1.", file=sys.stderr)
        sys.exit(1)

    print("[main] Step 3 - Building TF-IDF representation ...")
    summary = run_step3_tfidf(
        subset_csv_path=args.subset_csv,
        out_dir=args.out_dir,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
    )

    print("[main] Step 3 summary:")
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("  DONE. Output files:")
    print(f"    {os.path.join(args.out_dir, 'step3_tfidf_matrix.npz')}")
    print(f"    {os.path.join(args.out_dir, 'step3_labels.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step3_tfidf_vectorizer.joblib')}")
    print(f"    {os.path.join(args.out_dir, 'step3_vocabulary.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step3_summary.json')}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()