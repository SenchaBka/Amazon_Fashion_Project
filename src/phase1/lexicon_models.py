"""Lexicon-based sentiment prediction using VADER and TextBlob."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def _vader_label(compound: float) -> str:
    """Map VADER compound score to a three-class label.

    >= 0.05  → Positive
    <= -0.05 → Negative
    else     → Neutral
    """
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def _textblob_label(polarity: float) -> str:
    """Map TextBlob polarity score to a three-class label.

    > 0  → Positive
    < 0  → Negative
    == 0 → Neutral
    """
    if polarity > 0:
        return "Positive"
    if polarity < 0:
        return "Negative"
    return "Neutral"


def run_vader(texts: pd.Series) -> pd.Series:
    """Run VADER on each text and return a Series of sentiment labels.

    Uses compound score with thresholds ±0.05 per standard VADER guidance.
    """
    print(f"[lexicon] Running VADER on {len(texts):,} texts ...")
    analyzer = SentimentIntensityAnalyzer()
    labels = [_vader_label(analyzer.polarity_scores(str(t))["compound"]) for t in texts]
    print("[lexicon] VADER complete.")
    return pd.Series(labels, index=texts.index, name="vader_pred")


def run_textblob(texts: pd.Series) -> pd.Series:
    """Run TextBlob on each text and return a Series of sentiment labels.

    Uses TextBlob.sentiment.polarity; positive/negative threshold at zero.
    """
    print(f"[lexicon] Running TextBlob on {len(texts):,} texts ...")
    labels = [_textblob_label(TextBlob(str(t)).sentiment.polarity) for t in texts]
    print("[lexicon] TextBlob complete.")
    return pd.Series(labels, index=texts.index, name="textblob_pred")
