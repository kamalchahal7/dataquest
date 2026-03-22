"""
create_model_pkl.py — One-time script to train and serialize both models.

Trains:
  1. A stress classifier (Low/Medium/High) using combined_wellness_recommender.py
  2. A sleep quality classifier (Poor/Moderate/Good)

Run once:
    python create_model_pkl.py

Output: model.pkl (loaded by predictor.py at startup for fast serving)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from combined_wellness_recommender import (
    build_sleep_model,
    build_stress_model,
    prepare_sleep_data,
    prepare_stress_data,
    evaluate_sleep_model_cv,
    evaluate_stress_model_cv,
)
def _extract_feature_importances(pipeline, X: pd.DataFrame) -> pd.Series:
    """Map a fitted pipeline's feature importances back to interpretable names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in set(num_cols)]

    try:
        ohe_names = list(
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(cat_cols)
        )
    except Exception:
        ohe_names = []

    all_names = num_cols + ohe_names
    importances = pipeline.steps[-1][1].feature_importances_
    n = min(len(all_names), len(importances))
    return pd.Series(importances[:n], index=all_names[:n]).sort_values(ascending=False)


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _build_sleep_metrics(
    sleep_pipeline,
    sleep_X: pd.DataFrame,
    sleep_y: pd.Series,
    sleep_X_test: pd.DataFrame,
    sleep_y_test: pd.Series,
) -> dict:
    cv_results = evaluate_sleep_model_cv(sleep_pipeline, sleep_X, sleep_y)

    test_labels = sleep_pipeline.predict(sleep_X_test)

    return {
        "cv_classification": {
            "accuracy_mean": _round_metric(cv_results["accuracy_mean"]),
            "accuracy_std": _round_metric(cv_results["accuracy_std"]),
            "precision_macro_mean": _round_metric(cv_results["precision_mean"]),
            "precision_macro_std": _round_metric(cv_results["precision_std"]),
            "recall_macro_mean": _round_metric(cv_results["recall_mean"]),
            "recall_macro_std": _round_metric(cv_results["recall_std"]),
            "f1_macro_mean": _round_metric(cv_results["f1_macro_mean"]),
            "f1_macro_std": _round_metric(cv_results["f1_macro_std"]),
            "f1_weighted_mean": _round_metric(cv_results["f1_weighted_mean"]),
            "f1_weighted_std": _round_metric(cv_results["f1_weighted_std"]),
            "train_accuracy_mean": _round_metric(cv_results["train_accuracy_mean"]),
        },
        "test_classification": {
            "accuracy": _round_metric(accuracy_score(sleep_y_test, test_labels)),
            "precision_macro": _round_metric(
                precision_score(sleep_y_test, test_labels, average="macro", zero_division=0)
            ),
            "recall_macro": _round_metric(
                recall_score(sleep_y_test, test_labels, average="macro", zero_division=0)
            ),
            "f1_macro": _round_metric(
                f1_score(sleep_y_test, test_labels, average="macro", zero_division=0)
            ),
            "f1_weighted": _round_metric(
                f1_score(sleep_y_test, test_labels, average="weighted", zero_division=0)
            ),
        },
    }


def _build_stress_metrics(
    stress_pipeline,
    stress_X: pd.DataFrame,
    stress_y: pd.Series,
    stress_X_test: pd.DataFrame,
    stress_y_test: pd.Series,
) -> dict:
    cv_results = evaluate_stress_model_cv(stress_pipeline, stress_X, stress_y)
    test_predictions = stress_pipeline.predict(stress_X_test)

    return {
        "cv_classification": {
            "accuracy_mean": _round_metric(cv_results["accuracy_mean"]),
            "accuracy_std": _round_metric(cv_results["accuracy_std"]),
            "precision_macro_mean": _round_metric(cv_results["precision_mean"]),
            "precision_macro_std": _round_metric(cv_results["precision_std"]),
            "recall_macro_mean": _round_metric(cv_results["recall_mean"]),
            "recall_macro_std": _round_metric(cv_results["recall_std"]),
            "f1_macro_mean": _round_metric(cv_results["f1_macro_mean"]),
            "f1_macro_std": _round_metric(cv_results["f1_macro_std"]),
            "f1_weighted_mean": _round_metric(cv_results["f1_weighted_mean"]),
            "f1_weighted_std": _round_metric(cv_results["f1_weighted_std"]),
            "train_accuracy_mean": _round_metric(cv_results["train_accuracy_mean"]),
        },
        "test_classification": {
            "accuracy": _round_metric(accuracy_score(stress_y_test, test_predictions)),
            "precision_macro": _round_metric(
                precision_score(stress_y_test, test_predictions, average="macro", zero_division=0)
            ),
            "recall_macro": _round_metric(
                recall_score(stress_y_test, test_predictions, average="macro", zero_division=0)
            ),
            "f1_macro": _round_metric(
                f1_score(stress_y_test, test_predictions, average="macro", zero_division=0)
            ),
            "f1_weighted": _round_metric(
                f1_score(stress_y_test, test_predictions, average="weighted", zero_division=0)
            ),
        },
    }


def main():
    print("Loading and preparing datasets...")
    sleep_X, sleep_y = prepare_sleep_data()
    stress_X, stress_y = prepare_stress_data()

    # ── Stress classifier ──────────────────────────────────────────────────────
    print("Training stress classifier...")
    stress_X_train, stress_X_test, stress_y_train, stress_y_test = train_test_split(
        stress_X, stress_y, test_size=0.20, random_state=42, stratify=stress_y
    )
    stress_pipeline = build_stress_model(stress_X)
    stress_pipeline.fit(stress_X_train, stress_y_train)

    # ── Sleep classifier ──────────────────────────────────────────────────────
    print("Training sleep quality model...")
    sleep_X_train, sleep_X_test, sleep_y_train, sleep_y_test = train_test_split(
        sleep_X, sleep_y, test_size=0.20, random_state=42
    )
    sleep_pipeline = build_sleep_model(sleep_X)
    sleep_pipeline.fit(sleep_X_train, sleep_y_train)

    print("Computing showcase metrics...")
    stress_metrics = _build_stress_metrics(
        stress_pipeline, stress_X, stress_y, stress_X_test, stress_y_test
    )
    sleep_metrics = _build_sleep_metrics(
        sleep_pipeline, sleep_X, sleep_y, sleep_X_test, sleep_y_test
    )

    # ── Feature importances ────────────────────────────────────────────────────
    print("Extracting feature importances...")
    stress_feat_imp = _extract_feature_importances(stress_pipeline, stress_X)
    sleep_feat_imp  = _extract_feature_importances(sleep_pipeline,  sleep_X)

    # ── Label maps ────────────────────────────────────────────────────────────
    # Classes are stored in model order for predictor label mapping.
    stress_classes  = list(stress_pipeline.named_steps["classifier"].classes_)
    stress_label_map = {lbl: i for i, lbl in enumerate(stress_classes)}
    stress_inv_label = {i: lbl for i, lbl in enumerate(stress_classes)}

    sleep_classes    = list(sleep_pipeline.named_steps["classifier"].classes_)
    sleep_label_map  = {lbl: i for i, lbl in enumerate(sleep_classes)}
    sleep_inv_label  = {i: lbl for i, lbl in enumerate(sleep_classes)}

    # ── Column-level defaults for hidden / derived fields ─────────────────────
    stress_medians = stress_X.median(numeric_only=True).to_dict()
    stress_modes   = {
        c: stress_X[c].mode().iloc[0]
        for c in stress_X.select_dtypes(exclude=["number"]).columns
    }
    sleep_medians  = sleep_X.median(numeric_only=True).to_dict()
    sleep_modes    = {
        c: sleep_X[c].mode().iloc[0]
        for c in sleep_X.select_dtypes(exclude=["number"]).columns
    }

    # ── Serialize ─────────────────────────────────────────────────────────────
    bundle = {
        "stress_model":               stress_pipeline,
        "stress_feature_names":       list(stress_X.columns),
        "stress_label_map":           stress_label_map,
        "stress_inv_label":           stress_inv_label,
        "stress_feature_importances": stress_feat_imp,
        "stress_metrics":             stress_metrics,

        "sleep_model":                sleep_pipeline,
        "sleep_feature_names":        list(sleep_X.columns),
        "sleep_label_map":            sleep_label_map,
        "sleep_inv_label":            sleep_inv_label,
        "sleep_feature_importances":  sleep_feat_imp,
        "sleep_metrics":              sleep_metrics,

        # Defaults for fields not exposed in the app UI
        "stress_X_medians":           stress_medians,
        "stress_X_modes":             stress_modes,
        "sleep_X_medians":            sleep_medians,
        "sleep_X_modes":              sleep_modes,
    }

    out_path = Path(__file__).parent / "model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved model.pkl -> {out_path.resolve()}")
    print(f"  Stress classes  : {stress_classes}")
    print(f"  Stress features : {list(stress_X.columns)}")
    print(f"  Sleep features  : {list(sleep_X.columns)}")
    print(f"  Stress CV Acc   : {stress_metrics['cv_classification']['accuracy_mean']:.4f}")
    print(f"  Stress Test F1  : {stress_metrics['test_classification']['f1_macro']:.4f}")
    print(f"  Sleep CV F1     : {sleep_metrics['cv_classification']['f1_macro_mean']:.4f}")
    print(f"  Sleep Test Acc  : {sleep_metrics['test_classification']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
