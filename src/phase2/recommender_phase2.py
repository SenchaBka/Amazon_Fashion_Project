"""Phase 2 Step 7 utilities – Part A: review-based rating enhancement.

Adjusts original star ratings using sentiment labels derived in Step 1:

    Positive  →  original_rating + 0.25
    Neutral   →  original_rating  (no change)
    Negative  →  original_rating - 0.25

All adjusted values are clamped to the valid Amazon rating range [1.0, 5.0].

This is a lightweight, rule-based approach inspired by collaborative-filtering
ideas: when review text signals stronger (or weaker) satisfaction than the raw
star rating alone, the adjusted score better reflects the true user preference.
"""

import json
import os

import pandas as pd

# ── Adjustment rule ────────────────────────────────────────────────────────────
SENTIMENT_ADJUSTMENT: dict[str, float] = {
    "Positive": 0.25,
    "Neutral": 0.0,
    "Negative": -0.25,
}

RATING_MIN = 1.0
RATING_MAX = 5.0


def run_step7_recommender(
    subset_csv_path: str,
    out_dir: str,
) -> dict:
    """Apply sentiment-based rating adjustment and persist results + summary.

    Parameters
    ----------
    subset_csv_path:
        Path to the Step 1 labeled subset CSV.  Must contain columns
        ``overall`` (raw rating), ``sentiment``, and ``text``.
    out_dir:
        Directory where output artifacts are written.

    Returns
    -------
    dict
        Summary statistics suitable for inclusion in ``step7_summary.json``.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[recommender] Loading subset from '{subset_csv_path}' ...")
    df = pd.read_csv(subset_csv_path)

    required = {"overall", "sentiment", "text"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Required columns missing from subset CSV: {missing}")

    # Coerce types and drop unusable rows
    df = df.dropna(subset=["overall", "sentiment"]).copy()
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["overall"]).reset_index(drop=True)
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    print(f"[recommender] Processing {len(df)} reviews ...")

    # Apply adjustment rule
    df["adjustment"] = df["sentiment"].map(SENTIMENT_ADJUSTMENT).fillna(0.0)
    df["adjusted_rating"] = (
        (df["overall"] + df["adjustment"]).clip(RATING_MIN, RATING_MAX)
    )

    # ── Save detailed results CSV ──────────────────────────────────────────────
    results_df = pd.DataFrame(
        {
            "review_id": df.index,
            "review_text": df["text"],
            "original_rating": df["overall"],
            "sentiment_label": df["sentiment"],
            "adjustment": df["adjustment"],
            "adjusted_rating": df["adjusted_rating"],
        }
    )

    results_path = os.path.join(out_dir, "step7_recommender_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[recommender] Saved: {results_path}")

    # ── Save summary JSON ──────────────────────────────────────────────────────
    sentiment_counts = {
        k: int(v) for k, v in df["sentiment"].value_counts().to_dict().items()
    }

    summary = {
        "total_reviews_processed": int(len(df)),
        "mean_original_rating": round(float(df["overall"].mean()), 4),
        "mean_adjusted_rating": round(float(df["adjusted_rating"].mean()), 4),
        "min_adjusted_rating": round(float(df["adjusted_rating"].min()), 4),
        "max_adjusted_rating": round(float(df["adjusted_rating"].max()), 4),
        "sentiment_counts": sentiment_counts,
        "adjustment_rule": SENTIMENT_ADJUSTMENT,
        "rating_clamp": [RATING_MIN, RATING_MAX],
    }

    summary_path = os.path.join(out_dir, "step7_recommender_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[recommender] Saved: {summary_path}")

    return summary
