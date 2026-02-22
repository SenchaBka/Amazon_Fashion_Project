"""Entrypoint for Phase 1 lexicon sentiment analysis (Steps 3, 4, 6).

Run with:
    python -m src.run_phase1_lexicon --data_path data/AMAZON_FASHION.json --seed 42 --sample_size 1000
"""

import argparse
import os
import sys

from src.io_utils import load_jsonl
from src.preprocess_lexicon import prepare_dataframe, sample_balanced
from src.lexicon_models import run_vader, run_textblob
from src.evaluate import (
    compute_metrics,
    compute_confusion,
    save_outputs,
    print_metrics_table,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1 – Lexicon Sentiment (VADER + TextBlob)")
    parser.add_argument(
        "--data_path",
        default="data/AMAZON_FASHION.json",
        help="Path to JSONL dataset (.json or .jsonl). Default: data/AMAZON_FASHION.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of reviews to sample. Default: 1000",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs",
        help="Directory for output artifacts. Default: outputs",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 60)
    print("  PHASE 1 - LEXICON SENTIMENT ANALYSIS")
    print("=" * 60)
    print(f"  data_path   : {args.data_path}")
    print(f"  seed        : {args.seed}")
    print(f"  sample_size : {args.sample_size}")
    print(f"  out_dir     : {args.out_dir}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 1 – Load dataset
    # ------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        # Try swapping extension between .json and .jsonl
        alt = args.data_path.replace(".json", ".jsonl") if args.data_path.endswith(".json") \
            else args.data_path.replace(".jsonl", ".json")
        if os.path.exists(alt):
            print(f"[main] '{args.data_path}' not found; using '{alt}' instead.")
            args.data_path = alt
        else:
            print(f"[main] ERROR: dataset not found at '{args.data_path}' or '{alt}'.", file=sys.stderr)
            sys.exit(1)

    raw_df = load_jsonl(args.data_path)

    # ------------------------------------------------------------------
    # Step 2 – Preprocess and label (Step 4 in Phase 1)
    # ------------------------------------------------------------------
    print("\n[main] Step 4 - Preprocessing ...")
    clean_df = prepare_dataframe(raw_df)

    # ------------------------------------------------------------------
    # Step 3 – Sample 1000 labeled reviews (Step 5)
    # ------------------------------------------------------------------
    print(f"\n[main] Step 5 - Sampling {args.sample_size:,} reviews ...")
    sample_df = sample_balanced(clean_df, n=args.sample_size, seed=args.seed)

    # ------------------------------------------------------------------
    # Step 4 – Run lexicon models (Step 6)
    # ------------------------------------------------------------------
    print("\n[main] Step 6 - Running lexicon models ...")
    sample_df = sample_df.copy()
    sample_df["vader_pred"] = run_vader(sample_df["text"])
    sample_df["textblob_pred"] = run_textblob(sample_df["text"])

    # ------------------------------------------------------------------
    # Step 5 – Evaluate (Step 7)
    # ------------------------------------------------------------------
    print("\n[main] Step 7 - Evaluating models ...")
    y_true = sample_df["sentiment"]

    metrics_vader = compute_metrics(y_true, sample_df["vader_pred"], model_name="VADER")
    metrics_textblob = compute_metrics(y_true, sample_df["textblob_pred"], model_name="TextBlob")

    cm_vader = compute_confusion(y_true, sample_df["vader_pred"])
    cm_textblob = compute_confusion(y_true, sample_df["textblob_pred"])

    print_metrics_table(metrics_vader, metrics_textblob)

    print("\n[main] VADER confusion matrix:")
    print(cm_vader.to_string())

    print("\n[main] TextBlob confusion matrix:")
    print(cm_textblob.to_string())

    # ------------------------------------------------------------------
    # Step 6 – Save outputs
    # ------------------------------------------------------------------
    print(f"\n[main] Saving outputs to '{args.out_dir}/' ...")
    save_outputs(
        sample_df=sample_df,
        metrics_vader=metrics_vader,
        metrics_textblob=metrics_textblob,
        cm_vader=cm_vader,
        cm_textblob=cm_textblob,
        out_dir=args.out_dir,
    )

    print("\n" + "=" * 60)
    print("  DONE. Output files:")
    for fname in ["sample_1000.csv", "metrics.json", "comparison.csv",
                  "confusion_vader.csv", "confusion_textblob.csv"]:
        print(f"    {os.path.join(args.out_dir, fname)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
