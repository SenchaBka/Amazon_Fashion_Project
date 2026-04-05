"""Phase 2 Step 6.

Step 6: Compare Lexicon models (VADER, TextBlob) vs ML models (LogReg, SVM)
on the SAME test dataset (Phase 1's sample_1000.csv) for apples-to-apples comparison.

Run with:
    python -m src.phase2.run_phase2_step6
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.phase1.evaluate import compute_metrics

LABELS = ["Positive", "Neutral", "Negative"]


def _load_phase1_test_data(phase1_data_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load Phase 1 test data (sample_1000.csv) with true labels and text."""
    print(f"[step6] Loading Phase 1 test data from {phase1_data_path}...")
    df = pd.read_csv(phase1_data_path)
    
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Phase 1 data must have 'text' and 'sentiment' columns")
    
    text = df["text"]
    sentiment = df["sentiment"]
    
    print(f"[step6] Loaded {len(df)} samples from Phase 1")
    print(f"[step6] Distribution: {sentiment.value_counts().to_dict()}")
    
    return df, text, sentiment


def _load_vectorizer_and_predict(model_path: str, vectorizer_path: str, test_texts: pd.Series) -> np.ndarray:
    """Load model and vectorizer, then predict on test texts."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X_test_tfidf = vectorizer.transform(test_texts.values)
    return model.predict(X_test_tfidf)


def _calculate_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    """Calculate metrics using existing phase1 function."""
    metrics = compute_metrics(y_true, y_pred, model_name)
    return {k: v for k, v in metrics.items() if k != "model"}


def run_step6_comparison(
    phase1_data_path: str,
    step3_dir: str,
    step4_dir: str,
    step5_dir: str,
    out_dir: str,
) -> dict:
    """Compare all 4 models (VADER, TextBlob, LogReg, SVM) on Phase 1 test data."""
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n[step6] Loading Phase 1 test data for apples-to-apples comparison...")
    df, test_texts, test_labels = _load_phase1_test_data(phase1_data_path)
    
    results = {}
    confusions = {}
    
    # ===== VADER (From Phase 1) =====
    print("\n[step6] Using pre-computed VADER predictions from Phase 1...")
    if "vader_pred" in df.columns:
        vader_preds = df["vader_pred"]
    else:
        print("[step6] WARNING: vader_pred not found in Phase 1 data")
        vader_preds = pd.Series(["Positive"] * len(df))
    
    results["VADER"] = _calculate_metrics(test_labels, vader_preds, "VADER")
    confusions["VADER"] = confusion_matrix(test_labels, vader_preds, labels=LABELS)
    
    # ===== TextBlob (From Phase 1) =====
    print("[step6] Using pre-computed TextBlob predictions from Phase 1...")
    if "textblob_pred" in df.columns:
        textblob_preds = df["textblob_pred"]
    else:
        print("[step6] WARNING: textblob_pred not found in Phase 1 data")
        textblob_preds = pd.Series(["Positive"] * len(df))
    
    results["TextBlob"] = _calculate_metrics(test_labels, textblob_preds, "TextBlob")
    confusions["TextBlob"] = confusion_matrix(test_labels, textblob_preds, labels=LABELS)
    
    # ===== LogReg (Apply Phase 2 model) =====
    print("[step6] Loading LogReg model from Step 4 and predicting...")
    logreg_path = os.path.join(step4_dir, "step4_logreg_model.joblib")
    vectorizer_path = os.path.join(step3_dir, "step3_tfidf_vectorizer.joblib")
    
    if os.path.exists(logreg_path) and os.path.exists(vectorizer_path):
        logreg_preds = _load_vectorizer_and_predict(logreg_path, vectorizer_path, test_texts)
        results["LogReg"] = _calculate_metrics(test_labels, logreg_preds, "LogReg")
        confusions["LogReg"] = confusion_matrix(test_labels, logreg_preds, labels=LABELS)
    else:
        print(f"[step6] ERROR: LogReg model or vectorizer not found")
        results["LogReg"] = {"error": "Model or vectorizer not found"}
        confusions["LogReg"] = None
    
    # ===== SVM (Apply Phase 2 model) =====
    print("[step6] Loading SVM model from Step 5 and predicting...")
    svm_path = os.path.join(step5_dir, "step5_svm_model.joblib")
    
    if os.path.exists(svm_path) and os.path.exists(vectorizer_path):
        svm_preds = _load_vectorizer_and_predict(svm_path, vectorizer_path, test_texts)
        results["SVM"] = _calculate_metrics(test_labels, svm_preds, "SVM")
        confusions["SVM"] = confusion_matrix(test_labels, svm_preds, labels=LABELS)
    else:
        print(f"[step6] ERROR: SVM model or vectorizer not found")
        results["SVM"] = {"error": "Model or vectorizer not found"}
        confusions["SVM"] = None
    
    # ===== Save metrics =====
    metrics_path = os.path.join(out_dir, "step6_comparison_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    
    # ===== Create comparison table =====
    metrics_df = pd.DataFrame(results).T
    metrics_csv_path = os.path.join(out_dir, "step6_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv_path)
    
    print("\n[step6] Metrics comparison:")
    print(metrics_df)
    
    # ===== Create visualizations =====
    print("\n[step6] Creating visualizations...")
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Lexicon vs ML Models", fontsize=16, fontweight="bold")
    
    metrics_to_plot = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    colors = ["#FF6B6B", "#FFA500", "#4ECDC4", "#45B7D1"]
    model_names = ["VADER", "TextBlob", "LogReg", "SVM"]
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes.flatten()[idx]
        values = [results[model].get(metric, 0) for model in model_names]
        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    comparison_chart_path = os.path.join(out_dir, "step6_comparison_metrics.png")
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches="tight")
    print(f"[step6] Saved: {comparison_chart_path}")
    plt.close()
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Confusion Matrices: All Models", fontsize=16, fontweight="bold")
    
    for idx, (model_name, cm) in enumerate(confusions.items()):
        ax = axes.flatten()[idx]
        if cm is not None:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    cm_chart_path = os.path.join(out_dir, "step6_confusion_matrices.png")
    plt.savefig(cm_chart_path, dpi=300, bbox_inches="tight")
    print(f"[step6] Saved: {cm_chart_path}")
    plt.close()
    
    # ===== Summary =====
    summary = {
        "test_set_size": len(test_texts),
        "test_distribution": test_labels.value_counts().to_dict(),
        "models_compared": model_names,
        "metrics": results,
    }
    
    summary_path = os.path.join(out_dir, "step6_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Step 6 - Model Comparison (Apples-to-Apples)")
    parser.add_argument(
        "--phase1_data",
        default="outputs/sample_1000.csv",
        help="Path to Phase 1 test data (sample_1000.csv).",
    )
    parser.add_argument(
        "--step3_dir",
        default="outputs/phase 2/step 3",
        help="Directory containing Step 3 artifacts (TF-IDF vectorizer).",
    )
    parser.add_argument(
        "--step4_dir",
        default="outputs/phase 2/step 4",
        help="Directory containing Step 4 (LogReg) artifacts.",
    )
    parser.add_argument(
        "--step5_dir",
        default="outputs/phase 2/step 5",
        help="Directory containing Step 5 (SVM) artifacts.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/phase 2/step 6",
        help="Output directory for Step 6 artifacts.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 64)
    print("  PHASE 2 - STEP 6: APPLES-TO-APPLES MODEL COMPARISON")
    print("=" * 64)
    print(f"  phase1_data  : {args.phase1_data}")
    print(f"  step3_dir    : {args.step3_dir}")
    print(f"  step4_dir    : {args.step4_dir}")
    print(f"  step5_dir    : {args.step5_dir}")
    print(f"  out_dir      : {args.out_dir}")
    print("=" * 64 + "\n")

    if not os.path.isfile(args.phase1_data):
        print(f"[main] ERROR: Phase 1 data file not found at '{args.phase1_data}'.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.step3_dir):
        print(f"[main] ERROR: Step 3 directory not found at '{args.step3_dir}'.", file=sys.stderr)
        sys.exit(1)

    print("[main] Running Step 6 - Comparing all models on Phase 1 test data...")
    summary = run_step6_comparison(
        phase1_data_path=args.phase1_data,
        step3_dir=args.step3_dir,
        step4_dir=args.step4_dir,
        step5_dir=args.step5_dir,
        out_dir=args.out_dir,
    )

    print("\n[main] Step 6 summary:")
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("  DONE. Output files:")
    print(f"    {os.path.join(args.out_dir, 'step6_comparison_metrics.csv')}")
    print(f"    {os.path.join(args.out_dir, 'step6_comparison_metrics.json')}")
    print(f"    {os.path.join(args.out_dir, 'step6_comparison_metrics.png')}")
    print(f"    {os.path.join(args.out_dir, 'step6_confusion_matrices.png')}")
    print(f"    {os.path.join(args.out_dir, 'step6_summary.json')}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
