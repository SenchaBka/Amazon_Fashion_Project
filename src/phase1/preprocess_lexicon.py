"""Preprocessing tailored for lexicon-based sentiment models."""

import re
import numpy as np
import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH = str.maketrans({
    "\u200b": " ",
    "\u200c": " ",
    "\u200d": " ",
    "\ufeff": " ",
    "\u00ad": "",
})


def clean_text(s: str) -> str:
    """Lowercase, remove URLs, strip zero-width chars, and normalise whitespace.

    Deliberately keeps punctuation and emojis — both carry sentiment signal
    that VADER and TextBlob exploit.
    """
    s = str(s)
    s = s.translate(_ZERO_WIDTH)
    s = _URL_RE.sub("", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def label_sentiment(rating) -> str | float:
    """Map a numeric Amazon rating to a sentiment label.

    4-5  → Positive
    3    → Neutral
    1-2  → Negative
    Other → NaN (will be dropped)
    """
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


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Concatenate summary and reviewText into a single text field."""
    summary = df["summary"].fillna("").astype(str) if "summary" in df.columns else pd.Series("", index=df.index)
    review = df["reviewText"].fillna("").astype(str) if "reviewText" in df.columns else pd.Series("", index=df.index)
    return (summary + " " + review).str.strip()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply labelling, text building, and cleaning; return a tidy DataFrame.

    Drops rows with invalid ratings or empty text after cleaning.
    Returns columns: text, sentiment, overall.
    """
    print("[preprocess] Labelling sentiment from 'overall' column ...")
    if "overall" not in df.columns:
        raise KeyError("Column 'overall' (star rating) not found in dataset.")

    df = df.copy()
    df["sentiment"] = df["overall"].apply(label_sentiment)
    before = len(df)
    df = df.dropna(subset=["sentiment"])
    print(f"[preprocess] Dropped {before - len(df):,} rows with invalid ratings. Remaining: {len(df):,}")

    print("[preprocess] Building combined text field (summary + reviewText) ...")
    df["text_raw"] = build_text_field(df)

    print("[preprocess] Cleaning text ...")
    df["text"] = df["text_raw"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    print(f"[preprocess] Dropped {before - len(df):,} rows with empty text. Remaining: {len(df):,}")

    print("[preprocess] Sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    return df[["text", "sentiment", "overall"]].reset_index(drop=True)


def sample_balanced(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a stratified random sample of exactly n rows, preserving class proportions.

    If any class has too few rows to fill its proportional quota, it contributes
    all available rows and the remainder is filled from other classes proportionally.
    Falls back to a plain random sample if n >= len(df).
    """
    print(f"[preprocess] Sampling {n:,} rows (seed={seed}) ...")
    if n >= len(df):
        print(f"[preprocess] Requested sample ({n}) >= dataset size ({len(df)}). Returning full dataset.")
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    classes = df["sentiment"].unique()
    total = len(df)
    quota = {c: max(1, round(n * (df["sentiment"] == c).sum() / total)) for c in classes}

    # Adjust quota sum to exactly n
    diff = n - sum(quota.values())
    for c in sorted(classes):
        if diff == 0:
            break
        quota[c] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1

    parts = []
    rng = np.random.default_rng(seed)
    for c in classes:
        pool = df[df["sentiment"] == c]
        k = min(quota[c], len(pool))
        parts.append(pool.sample(n=k, random_state=rng.integers(0, 2**31)))

    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[preprocess] Sample sentiment distribution:\n{sample['sentiment'].value_counts().to_string()}")
    return sample
