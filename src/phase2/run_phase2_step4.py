"""Entrypoint for Phase 2 Step 4.

Step 4: Train and tune Logistic Regression using stratified 70% training split.

Run with:
    python -m src.phase2.run_phase2_step4
"""

import argparse
import json
import os
import sys

from src.phase2.model_phase2 import run_step4_logreg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Step 4 - Logistic Regression Training/Tuning")
    parser.add_argument(
        "--step3_dir",
        default="outputs/phase 2/step 3",
        help="Directory containing Step 3 artifacts.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/phase 2/step 4",
        help="Output directory for Step 4 artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of stratified folds for hyperparameter tuning.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 64)
    print("  PHASE 2 - STEP 4")
    print("=" * 64)
    print(f"  step3_dir : {args.step3_dir}")
    print(f"  out_dir   : {args.out_dir}")
    print(f"  seed      : {args.seed}")
    print(f"  cv_folds  : {args.cv_folds}")
    print("=" * 64 + "\n")

    if not os.path.isdir(args.step3_dir):
        print(f"[main] ERROR: Step 3 directory not found at '{args.step3_dir}'.", file=sys.stderr)
        sys.exit(1)
    if args.cv_folds < 2:
        print("[main] ERROR: --cv_folds must be >= 2.", file=sys.stderr)
        sys.exit(1)

    print("[main] Step 4 - Training + tuning Logistic Regression ...")
    summary = run_step4_logreg(
        step3_dir=args.step3_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )

    print("[main] Step 4 summary:")
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("  DONE. Output files:")
    print(f"    {os.path.join(args.out_dir, 'step4_logreg_model.joblib')}")
    print(f"    {os.path.join(args.out_dir, 'step4_best_params.json')}")
    print(f"    {os.path.join(args.out_dir, 'step4_metrics.json')}")
    print(f"    {os.path.join(args.out_dir, 'step4_confusion_matrix.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step4_test_predictions.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step4_cv_results_top10.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step4_summary.json')}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()