"""Phase 2 Step 5 utilities (SVM training/tuning)."""

import json
import os

import joblib
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

LABELS = ["Positive", "Neutral", "Negative"]
TRAIN_FRACTION = 0.70


def _distribution(y: pd.Series) -> dict:
    return {k: int(v) for k, v in y.value_counts().reindex(LABELS, fill_value=0).to_dict().items()}


def run_step5_svm(
    step3_dir: str,
    out_dir: str,
    seed: int = 42,
    cv_folds: int = 5,
) -> dict:
    """Train/tune SVM with a stratified 70% training split."""
    os.makedirs(out_dir, exist_ok=True)

    matrix_path = os.path.join(step3_dir, "step3_tfidf_matrix.npz")
    labels_path = os.path.join(step3_dir, "step3_labels.csv")

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"TF-IDF matrix file not found: {matrix_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Step 3 labels file not found: {labels_path}")

    X = sparse.load_npz(matrix_path)
    y_df = pd.read_csv(labels_path)
    if "sentiment" not in y_df.columns:
        raise KeyError("Column 'sentiment' not found in Step 3 labels file.")

    y = y_df["sentiment"].astype(str)
    if X.shape[0] != len(y):
        raise ValueError(f"Row mismatch between TF-IDF matrix ({X.shape[0]}) and labels ({len(y)}).")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=TRAIN_FRACTION,
        random_state=seed,
        stratify=y,
    )

    model = SVC(random_state=seed, kernel="linear")
    grid = {
        "C": [0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_precision": round(precision_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
    }

    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=LABELS),
        index=[f"actual_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    )

    model_path = os.path.join(out_dir, "step5_svm_model.joblib")
    params_path = os.path.join(out_dir, "step5_best_params.json")
    metrics_path = os.path.join(out_dir, "step5_metrics.json")
    cm_path = os.path.join(out_dir, "step5_confusion_matrix.csv")
    pred_path = os.path.join(out_dir, "step5_test_predictions.csv")
    cv_path = os.path.join(out_dir, "step5_cv_results_top10.csv")
    summary_path = os.path.join(out_dir, "step5_summary.json")

    joblib.dump(best_model, model_path)

    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(search.best_params_, fh, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    cm_df.to_csv(cm_path)

    pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
        }
    ).to_csv(pred_path, index=False)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(10)
    keep_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_C",
        "param_class_weight",
    ]
    cv_results[keep_cols].to_csv(cv_path, index=False)

    summary = {
        "step3_input_dir": step3_dir,
        "total_rows": X.shape[0],
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "train_fraction": TRAIN_FRACTION,
        "stratified_split": True,
        "seed": seed,
        "cv_folds": cv_folds,
        "train_sentiment_distribution": _distribution(y_train),
        "test_sentiment_distribution": _distribution(y_test),
        "best_params": search.best_params_,
        "best_cv_f1_macro": round(search.best_score_, 4),
        "holdout_metrics": metrics,
        "artifacts": {
            "model": model_path,
            "best_params": params_path,
            "metrics": metrics_path,
            "confusion_matrix": cm_path,
            "test_predictions": pred_path,
            "cv_results_top10": cv_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary