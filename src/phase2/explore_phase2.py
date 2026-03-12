"""Phase 2 dataset exploration utilities (Step 2)."""

import json
import os
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(v) -> float:
    return float(v) if pd.notna(v) else 0.0


def _save_series_csv(series: pd.Series, path: str, value_name: str) -> None:
    series.rename_axis("key").reset_index(name=value_name).to_csv(path, index=False)


def run_step2_exploration(df: pd.DataFrame, out_dir: str) -> dict:
    """Compute Step 2 exploration metrics and save plots/tables."""
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    rating_counts = df["overall"].value_counts().sort_index()
    sentiment_counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)

    reviews_per_product = (
        df.groupby("asin").size().sort_values(ascending=False)
        if "asin" in df.columns
        else pd.Series(dtype="int64")
    )
    reviews_per_user = (
        df.groupby("reviewerID").size().sort_values(ascending=False)
        if "reviewerID" in df.columns
        else pd.Series(dtype="int64")
    )

    q1, q3 = df["word_len"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    outlier_count = int(((df["word_len"] < lower) | (df["word_len"] > upper)).sum())

    text_norm = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    duplicate_text_count = int(text_norm.duplicated().sum())

    dedup_keys = [c for c in ["reviewerID", "asin", "overall", "text"] if c in df.columns]
    exact_duplicate_count = int(df.duplicated(subset=dedup_keys).sum()) if dedup_keys else 0

    summary = {
        "subset_rows": int(len(df)),
        "unique_products": int(df["asin"].nunique()) if "asin" in df.columns else 0,
        "unique_users": int(df["reviewerID"].nunique()) if "reviewerID" in df.columns else 0,
        "average_rating": round(_safe_float(df["overall"].mean()), 4),
        "median_rating": round(_safe_float(df["overall"].median()), 4),
        "sentiment_distribution": {k: int(v) for k, v in sentiment_counts.to_dict().items()},
        "rating_distribution": {str(k): int(v) for k, v in rating_counts.to_dict().items()},
        "word_length_stats": {
            "mean": round(_safe_float(df["word_len"].mean()), 4),
            "median": round(_safe_float(df["word_len"].median()), 4),
            "p95": round(_safe_float(df["word_len"].quantile(0.95)), 4),
            "p99": round(_safe_float(df["word_len"].quantile(0.99)), 4),
        },
        "outlier_thresholds_iqr": {"lower": round(_safe_float(lower), 4), "upper": round(_safe_float(upper), 4)},
        "outlier_review_count": outlier_count,
        "duplicate_normalized_text_count": duplicate_text_count,
        "exact_duplicate_count": exact_duplicate_count,
        "reviews_per_product": {
            "mean": round(_safe_float(reviews_per_product.mean()), 4) if len(reviews_per_product) else 0.0,
            "median": round(_safe_float(reviews_per_product.median()), 4) if len(reviews_per_product) else 0.0,
            "max": int(reviews_per_product.max()) if len(reviews_per_product) else 0,
        },
        "reviews_per_user": {
            "mean": round(_safe_float(reviews_per_user.mean()), 4) if len(reviews_per_user) else 0.0,
            "median": round(_safe_float(reviews_per_user.median()), 4) if len(reviews_per_user) else 0.0,
            "max": int(reviews_per_user.max()) if len(reviews_per_user) else 0,
        },
    }

    with open(os.path.join(out_dir, "step2_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _save_series_csv(rating_counts, os.path.join(out_dir, "rating_distribution.csv"), "count")
    _save_series_csv(sentiment_counts, os.path.join(out_dir, "sentiment_distribution.csv"), "count")

    if len(reviews_per_product):
        _save_series_csv(reviews_per_product.head(20), os.path.join(out_dir, "top_products_by_reviews.csv"), "count")
    if len(reviews_per_user):
        _save_series_csv(reviews_per_user.head(20), os.path.join(out_dir, "top_users_by_reviews.csv"), "count")

    plt.figure(figsize=(7, 4))
    rating_counts.plot(kind="bar")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rating_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    df["word_len"].plot(kind="hist", bins=60, log=True)
    plt.title("Review Word Length Distribution (log y)")
    plt.xlabel("Words per Review")
    plt.ylabel("Frequency (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "review_word_length_hist.png"), dpi=150)
    plt.close()

    if len(reviews_per_product):
        plt.figure(figsize=(7, 4))
        reviews_per_product.plot(kind="hist", bins=50, log=True)
        plt.title("Reviews per Product (log y)")
        plt.xlabel("Reviews per Product")
        plt.ylabel("Frequency (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "reviews_per_product_hist.png"), dpi=150)
        plt.close()

    if len(reviews_per_user):
        plt.figure(figsize=(7, 4))
        reviews_per_user.plot(kind="hist", bins=50, log=True)
        plt.title("Reviews per User (log y)")
        plt.xlabel("Reviews per User")
        plt.ylabel("Frequency (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "reviews_per_user_hist.png"), dpi=150)
        plt.close()

    return summary
