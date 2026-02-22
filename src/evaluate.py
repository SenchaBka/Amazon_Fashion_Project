"""Metrics computation and output artifact writers for lexicon model evaluation."""

import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

LABELS = ["Positive", "Neutral", "Negative"]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    """Compute accuracy, macro precision, recall, and F1 for a single model."""
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
    }


def compute_confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return a labelled 3x3 confusion matrix DataFrame (rows=actual, cols=predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def save_outputs(
    sample_df: pd.DataFrame,
    metrics_vader: dict,
    metrics_textblob: dict,
    cm_vader: pd.DataFrame,
    cm_textblob: pd.DataFrame,
    out_dir: str = "outputs",
) -> None:
    """Write all evaluation artifacts to out_dir.

    Files produced:
      sample_1000.csv          - text + label + predictions
      metrics.json             - accuracy/precision/recall/F1 for both models
      comparison.csv           - side-by-side metrics table
      confusion_vader.csv      - 3x3 confusion matrix for VADER
      confusion_textblob.csv   - 3x3 confusion matrix for TextBlob
    """
    os.makedirs(out_dir, exist_ok=True)

    sample_path = os.path.join(out_dir, "sample_1000.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"[evaluate] Saved sample → {sample_path}")

    metrics_all = [metrics_vader, metrics_textblob]
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_all, fh, indent=2)
    print(f"[evaluate] Saved metrics → {metrics_path}")

    comparison_path = os.path.join(out_dir, "comparison.csv")
    pd.DataFrame(metrics_all).to_csv(comparison_path, index=False)
    print(f"[evaluate] Saved comparison table → {comparison_path}")

    cm_v_path = os.path.join(out_dir, "confusion_vader.csv")
    cm_vader.to_csv(cm_v_path)
    print(f"[evaluate] Saved VADER confusion matrix → {cm_v_path}")

    cm_tb_path = os.path.join(out_dir, "confusion_textblob.csv")
    cm_textblob.to_csv(cm_tb_path)
    print(f"[evaluate] Saved TextBlob confusion matrix → {cm_tb_path}")


def print_metrics_table(metrics_vader: dict, metrics_textblob: dict) -> None:
    """Print a formatted side-by-side metrics comparison to stdout."""
    df = pd.DataFrame([metrics_vader, metrics_textblob]).set_index("model")
    print("\n" + "=" * 55)
    print("  METRICS COMPARISON")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55)
