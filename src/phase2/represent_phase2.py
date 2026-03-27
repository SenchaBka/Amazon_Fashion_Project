"""Phase 2 Step 3 utilities (preprocessing + TF-IDF representation)."""

import json
import os
import re

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.phase2.preprocess_phase2 import clean_text

_WS_RE = re.compile(r"\s+")
LABELS = ["Positive", "Neutral", "Negative"]


def _normalize_for_tfidf(s: str) -> str:
    """Apply simple normalization for TF-IDF input text."""
    s = clean_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def run_step3_tfidf(
    subset_csv_path: str,
    out_dir: str,
    max_features: int = 20000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    """Build TF-IDF representation from Step 1 subset and save artifacts."""
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(subset_csv_path)
    if "text" not in df.columns:
        raise KeyError("Column 'text' not found in Step 1 subset file.")
    if "sentiment" not in df.columns:
        raise KeyError("Column 'sentiment' not found in Step 1 subset file.")

    original_rows = int(len(df))

    work = df.copy()
    work["text_step3"] = work["text"].fillna("").astype(str).apply(_normalize_for_tfidf)
    work = work[work["text_step3"].str.len() > 0].copy()
    work = work[work["sentiment"].isin(LABELS)].copy()
    work = work.reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(work["text_step3"])

    matrix_path = os.path.join(out_dir, "step3_tfidf_matrix.npz")
    labels_path = os.path.join(out_dir, "step3_labels.csv")
    vectorizer_path = os.path.join(out_dir, "step3_tfidf_vectorizer.joblib")
    vocab_path = os.path.join(out_dir, "step3_vocabulary.csv")
    summary_path = os.path.join(out_dir, "step3_summary.json")

    sparse.save_npz(matrix_path, X)

    label_cols = [c for c in ["sentiment", "overall", "asin", "reviewerID", "unixReviewTime"] if c in work.columns]
    work[label_cols].to_csv(labels_path, index=False)

    joblib.dump(vectorizer, vectorizer_path)

    pd.DataFrame({"feature": vectorizer.get_feature_names_out()}).to_csv(vocab_path, index=False)

    summary = {
        "input_subset_path": subset_csv_path,
        "rows_input": original_rows,
        "rows_after_step3_preprocessing": int(len(work)),
        "matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        "tfidf_config": {
            "max_features": int(max_features),
            "min_df": int(min_df),
            "max_df": float(max_df),
            "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
            "sublinear_tf": True,
        },
        "sentiment_distribution": {
            k: int(v)
            for k, v in work["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_dict().items()
        },
        "artifacts": {
            "tfidf_matrix": matrix_path,
            "labels": labels_path,
            "vectorizer": vectorizer_path,
            "vocabulary": vocab_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary