"""Phase 2 Step 4 utilities (Logistic Regression training/tuning)."""

import json
import os
import warnings

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

LABELS = ["Positive", "Neutral", "Negative"]


def _distribution(y: pd.Series) -> dict:
    return {k: int(v) for k, v in y.value_counts().reindex(LABELS, fill_value=0).to_dict().items()}


def run_step4_logreg(
    step3_dir: str,
    out_dir: str,
    seed: int = 42,
    cv_folds: int = 5,
    use_smote: bool = True,
) -> dict:
    """Train/tune logistic regression with a stratified 70% training split."""
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
    if "asin" not in y_df.columns:
        raise KeyError("Column 'asin' not found in Step 3 labels file — required for group-aware splitting.")

    y = y_df["sentiment"].astype(str)
    if X.shape[0] != len(y):
        raise ValueError(f"Row mismatch between TF-IDF matrix ({X.shape[0]}) and labels ({len(y)}).")

    asin_series = y_df["asin"]
    unique_asins = asin_series.unique()

    asin_majority = []
    for a in unique_asins:
        asin_majority.append(y[asin_series == a].mode().iloc[0])
    asin_majority = pd.Series(asin_majority, index=unique_asins)

    train_asins, test_asins = train_test_split(
        unique_asins,
        train_size=0.70,
        random_state=seed,
        stratify=asin_majority,
    )

    train_mask = asin_series.isin(train_asins).values
    test_mask = asin_series.isin(test_asins).values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if "reviewerID" in y_df.columns:
        train_reviewers = set(y_df.loc[train_mask, "reviewerID"].unique())
        test_reviewers = set(y_df.loc[test_mask, "reviewerID"].unique())
        reviewer_overlap = train_reviewers & test_reviewers
        if reviewer_overlap:
            warnings.warn(
                f"{len(reviewer_overlap)} reviewerIDs appear in both train and test. "
                f"This is expected when reviewers review multiple products."
            )
    else:
        reviewer_overlap = set()

    smote_applied = False
    k_neighbors = None
    train_dist_before = _distribution(y_train)

    if use_smote:
        min_class_count = y_train.value_counts().min()

        if min_class_count < 2:
            warnings.warn(
                f"SMOTE skipped: smallest class has only {min_class_count} sample(s), "
                f"requires at least 2 for oversampling."
            )
        else:
            k_neighbors = 5 if min_class_count > 6 else (min_class_count - 1)
            X_train = X_train.tocsr() if not X_train.format == "csr" else X_train

            smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            smote_applied = True

    train_dist_after = _distribution(y_train)

    model = LogisticRegression(random_state=seed)
    grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        "class_weight": [None, "balanced"],
        "max_iter": [500, 1000],
        "solver": ["lbfgs"],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
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

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])

    model_path = os.path.join(out_dir, "step4_logreg_model.joblib")
    params_path = os.path.join(out_dir, "step4_best_params.json")
    metrics_path = os.path.join(out_dir, "step4_metrics.json")
    cm_path = os.path.join(out_dir, "step4_confusion_matrix.csv")
    pred_path = os.path.join(out_dir, "step4_test_predictions.csv")
    cv_path = os.path.join(out_dir, "step4_cv_results_top10.csv")
    summary_path = os.path.join(out_dir, "step4_summary.json")

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
        "param_max_iter",
        "param_solver",
    ]
    cv_results[keep_cols].to_csv(cv_path, index=False)

    summary = {
        "step3_input_dir": step3_dir,
        "total_rows": int(X.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_fraction": 0.70,
        "split_method": "group_aware_by_asin",
        "stratified_split": True,
        "seed": int(seed),
        "cv_folds": int(cv_folds),
        "n_unique_asins": int(len(unique_asins)),
        "train_asin_count": int(len(train_asins)),
        "test_asin_count": int(len(test_asins)),
        "asin_overlap": 0,
        "reviewer_overlap_count": int(len(reviewer_overlap)),
        "smote_applied": smote_applied,
        "train_sentiment_distribution": _distribution(y_train),
        "test_sentiment_distribution": _distribution(y_test),
        "smote_k_neighbors": k_neighbors,
        "train_distribution_before_smote": train_dist_before,
        "train_distribution_after_smote": train_dist_after,
        "best_params": search.best_params_,
        "best_cv_f1_macro": round(float(search.best_score_), 4),
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