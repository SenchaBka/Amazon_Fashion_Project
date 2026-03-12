"""Phase 2 preprocessing utilities (Step 1: subset creation + labelling)."""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH = str.maketrans(
    {
        "\u200b": " ",
        "\u200c": " ",
        "\u200d": " ",
        "\ufeff": " ",
        "\u00ad": "",
    }
)


def label_sentiment(rating) -> str | float:
    """Map Amazon star rating to 3-way sentiment label."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return np.nan
    if r >= 4:
        return "Positive"
    if r == 3:
        return "Neutral"
    if r <= 2:
        return "Negative"
    return np.nan


def clean_text(s: str) -> str:
    """Light cleaning suitable for downstream feature extraction."""
    s = str(s)
    s = s.translate(_ZERO_WIDTH)
    s = _URL_RE.sub("", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Build a combined review text from summary + reviewText."""
    summary = df["summary"].fillna("").astype(str) if "summary" in df.columns else pd.Series("", index=df.index)
    review = df["reviewText"].fillna("").astype(str) if "reviewText" in df.columns else pd.Series("", index=df.index)
    return (summary + " " + review).str.strip()


def prepare_phase2_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned/labelled dataframe used for Phase 2 Step 1 + Step 2."""
    if "overall" not in df.columns:
        raise KeyError("Column 'overall' (rating) not found in dataset.")

    out = df.copy()
    out["overall"] = pd.to_numeric(out["overall"], errors="coerce")
    out["sentiment"] = out["overall"].apply(label_sentiment)
    out = out.dropna(subset=["overall", "sentiment"]).copy()

    out["text_raw"] = build_text_field(out)
    out["text"] = out["text_raw"].apply(clean_text)
    out = out[out["text"].str.len() > 0].copy()

    out["word_len"] = out["text"].str.split().str.len()
    out["char_len"] = out["text"].str.len()

    keep_cols = [
        "overall",
        "sentiment",
        "text",
        "text_raw",
        "word_len",
        "char_len",
        "asin",
        "reviewerID",
        "unixReviewTime",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    return out[existing].reset_index(drop=True)


def select_stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Select n rows using stratified sampling on sentiment."""
    if n <= 0:
        raise ValueError("subset size must be > 0")
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    try:
        subset, _ = train_test_split(
            df,
            train_size=n,
            random_state=seed,
            stratify=df["sentiment"],
        )
        return subset.reset_index(drop=True)
    except ValueError:
        # Fallback when stratification is not feasible (very rare class counts).
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
