"""Entrypoint for Phase 2 Step 7.

Step 7 covers three tasks using the same Phase 2 subset (~2000 reviews) that
was created in Step 1:

  Part A – Review-based rating enhancement
      Adjusts each original star rating by ±0.25 based on sentiment label
      (Positive / Neutral / Negative), then clamps to [1.0, 5.0].

  Part B – LLM summarization
      Selects 10 reviews longer than 100 words and summarises each to ~50
      words using a local Hugging Face summarization pipeline.

  Part C – LLM customer-service response generation
      Finds one question-like review and generates a polite, professional
      reply using a local Hugging Face text-to-text pipeline.

Run with:
    python -m src.phase2.run_phase2_step7

Skip LLM parts (if transformers is not installed):
    python -m src.phase2.run_phase2_step7 --skip_llm

Override models:
    python -m src.phase2.run_phase2_step7 \\
        --summarization_model sshleifer/distilbart-cnn-12-6 \\
        --generation_model google/flan-t5-base
"""

import argparse
import json
import os
import sys

from src.phase2.recommender_phase2 import run_step7_recommender
from src.phase2.llm_phase2 import (
    GENERATION_MODEL,
    SUMMARIZATION_MODEL,
    run_step7_response_generation,
    run_step7_summarization,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2 Step 7 – Rating Enhancement + LLM Summarization + Response Generation"
    )
    parser.add_argument(
        "--subset_csv",
        default="outputs/phase 2/step 1/step1_subset_labeled.csv",
        help="Path to the Step 1 labeled subset CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/phase 2/step 7",
        help="Output directory for Step 7 artifacts.",
    )
    parser.add_argument(
        "--summarization_model",
        default=SUMMARIZATION_MODEL,
        help="Hugging Face model name for summarization (Part B).",
    )
    parser.add_argument(
        "--generation_model",
        default=GENERATION_MODEL,
        help="Hugging Face model name for response generation (Part C).",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help=(
            "Skip Parts B and C (LLM tasks). "
            "Use this flag if the 'transformers' package is not installed."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 64)
    print("  PHASE 2 - STEP 7")
    print("=" * 64)
    print(f"  subset_csv          : {args.subset_csv}")
    print(f"  out_dir             : {args.out_dir}")
    print(f"  summarization_model : {args.summarization_model}")
    print(f"  generation_model    : {args.generation_model}")
    print(f"  skip_llm            : {args.skip_llm}")
    print("=" * 64 + "\n")

    if not os.path.isfile(args.subset_csv):
        print(
            f"[main] ERROR: Subset CSV not found at '{args.subset_csv}'.\n"
            "[main] Run Step 1 first:  python -m src.phase2.run_phase2_steps12",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Part A: Rating enhancement ─────────────────────────────────────────────
    print("[main] Part A – Running review-based rating enhancement ...")
    recommender_summary = run_step7_recommender(
        subset_csv_path=args.subset_csv,
        out_dir=args.out_dir,
    )
    print("[main] Part A complete. Key statistics:")
    print(f"       mean original rating  : {recommender_summary['mean_original_rating']}")
    print(f"       mean adjusted rating  : {recommender_summary['mean_adjusted_rating']}")

    # ── Part B: LLM summarization ──────────────────────────────────────────────
    if args.skip_llm:
        print("\n[main] Part B – SKIPPED (--skip_llm flag set).")
        summarization_summary: dict = {"skipped": True}
    else:
        print("\n[main] Part B – Selecting long reviews and generating summaries ...")
        try:
            summarization_summary = run_step7_summarization(
                subset_csv_path=args.subset_csv,
                out_dir=args.out_dir,
                model_name=args.summarization_model,
            )
            print(
                f"[main] Part B complete. "
                f"Summarised {summarization_summary['reviews_selected']} reviews."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[main] Part B FAILED: {exc}", file=sys.stderr)
            summarization_summary = {"error": str(exc)}

    # ── Part C: LLM response generation ───────────────────────────────────────
    if args.skip_llm:
        print("[main] Part C – SKIPPED (--skip_llm flag set).")
        generation_summary: dict = {"skipped": True}
    else:
        print("\n[main] Part C – Generating customer-service response ...")
        try:
            generation_summary = run_step7_response_generation(
                subset_csv_path=args.subset_csv,
                out_dir=args.out_dir,
                model_name=args.generation_model,
            )
            print("[main] Part C complete.")
            print(f"       review  : {generation_summary.get('review_preview', '')}")
            print(f"       response: {generation_summary.get('response_preview', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[main] Part C FAILED: {exc}", file=sys.stderr)
            generation_summary = {"error": str(exc)}

    # ── Write overall Step 7 summary ───────────────────────────────────────────
    step7_summary = {
        "subset_csv": args.subset_csv,
        "out_dir": args.out_dir,
        "part_a_recommender": recommender_summary,
        "part_b_summarization": summarization_summary,
        "part_c_response_generation": generation_summary,
    }
    summary_path = os.path.join(args.out_dir, "step7_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(step7_summary, fh, indent=2)
    print(f"\n[main] Step 7 summary written to: {summary_path}")

    print("\n" + "=" * 64)
    print("  DONE. Output files:")
    print(f"    {os.path.join(args.out_dir, 'step7_recommender_results.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step7_recommender_summary.json')}")
    if not args.skip_llm:
        print(f"    {os.path.join(args.out_dir, 'step7_long_reviews.csv')}")
        print(f"    {os.path.join(args.out_dir, 'step7_summaries.csv')}")
        print(f"    {os.path.join(args.out_dir, 'step7_question_review.csv')}")
        print(f"    {os.path.join(args.out_dir, 'step7_generated_response.txt')}")
    print(f"    {summary_path}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
