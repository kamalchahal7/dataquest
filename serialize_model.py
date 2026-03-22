"""
serialize_model.py  —  Train models from combined_wellness_recommender.py
and save them to model.pkl in the format expected by predictor.py.

Run once (both CSV datasets must be present in the same directory):
    python serialize_model.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from combined_wellness_recommender import (
    prepare_sleep_data,
    prepare_stress_data,
    build_sleep_model,
    build_stress_model,
    oversample_low_class,
    STRESS_TARGET_ORDER,
)

# ── Sleep quality thresholds (dataset uses 1–10 scale) ────────────────────────
SLEEP_LABELS      = ["Poor", "Moderate", "Good"]
SLEEP_POOR_MAX    = 6.0    # score < 6.0  → Poor
SLEEP_MOD_MAX     = 7.5    # 6.0 ≤ score < 7.5 → Moderate  (≥ 7.5 → Good)
SLEEP_CLASS_CENTERS = {"Poor": 4.0, "Moderate": 6.75, "Good": 8.75}


class SleepClassifierWrapper:
    """
    Wraps a GradientBoosting regression Pipeline to expose a
    classifier-compatible API (predict / predict_proba).
    Soft probabilities are computed via inverse-distance weighting
    over fixed class-centre points on the 1-10 quality scale.
    """

    def __init__(self, regressor):
        self.regressor = regressor
        self.labels    = SLEEP_LABELS
        self.classes_  = np.array(SLEEP_LABELS)

    def _score_to_class(self, score: float) -> str:
        if score < SLEEP_POOR_MAX:
            return "Poor"
        if score < SLEEP_MOD_MAX:
            return "Moderate"
        return "Good"

    def _score_to_proba(self, score: float) -> np.ndarray:
        centers = np.array([SLEEP_CLASS_CENTERS[lbl] for lbl in self.labels])
        dists   = np.abs(score - centers)
        inv     = 1.0 / (dists + 0.5)   # +0.5 avoids division by zero
        return inv / inv.sum()

    def predict(self, X):
        scores = self.regressor.predict(X)
        return np.array([self._score_to_class(s) for s in scores])

    def predict_proba(self, X):
        scores = self.regressor.predict(X)
        return np.array([self._score_to_proba(s) for s in scores])


# ── Feature importance helpers ─────────────────────────────────────────────────
def _map_importances(pipeline, original_feature_names: list) -> pd.Series:
    """
    Sum ColumnTransformer-transformed feature importances back to original
    input column names.  Falls back to uniform importances on any error.
    """
    fallback = pd.Series(
        np.ones(len(original_feature_names)) / len(original_feature_names),
        index=original_feature_names,
    ).sort_values(ascending=False)

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        tf_names     = list(preprocessor.get_feature_names_out())
    except Exception:
        return fallback

    final_key = list(pipeline.named_steps.keys())[-1]
    estimator  = pipeline.named_steps[final_key]
    if not hasattr(estimator, "feature_importances_"):
        return fallback

    raw_imp = estimator.feature_importances_
    imp_map = {}
    for orig in original_feature_names:
        total = 0.0
        for i, tname in enumerate(tf_names):
            # ColumnTransformer prefixes: "num__FeatureName" / "cat__FeatureName_Value"
            if tname == f"num__{orig}" or tname.startswith(f"cat__{orig}_") or tname == f"cat__{orig}":
                total += raw_imp[i]
        imp_map[orig] = total

    series = pd.Series(imp_map).sort_values(ascending=False)
    total  = series.sum()
    return series / total if total > 0 else series


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading datasets...")
    sleep_X, sleep_y   = prepare_sleep_data()
    stress_X, stress_y = prepare_stress_data()

    print("Training models...")
    sleep_pipeline  = build_sleep_model(sleep_X)
    stress_pipeline = build_stress_model(stress_X)

    # Mirror the train/test split used in combined_wellness_recommender.main()
    sleep_X_train, _, sleep_y_train, _ = train_test_split(
        sleep_X, sleep_y, test_size=0.20, random_state=42
    )
    stress_X_train, _, stress_y_train, _ = train_test_split(
        stress_X, stress_y, test_size=0.20, random_state=42, stratify=stress_y
    )

    sleep_pipeline.fit(sleep_X_train, sleep_y_train)

    stress_X_res, stress_y_res = oversample_low_class(stress_X_train, stress_y_train)
    stress_pipeline.fit(stress_X_res, stress_y_res)
    print("Models trained.")

    # Wrap sleep regressor so it looks like a classifier
    sleep_classifier = SleepClassifierWrapper(sleep_pipeline)

    # Feature names (raw input column names that each Pipeline expects)
    stress_feature_names = list(stress_X.columns)
    sleep_feature_names  = list(sleep_X.columns)

    # Label maps
    stress_label_map = {lbl: i for i, lbl in enumerate(STRESS_TARGET_ORDER)}
    stress_inv_label = {i: lbl for i, lbl in enumerate(STRESS_TARGET_ORDER)}
    sleep_label_map  = {lbl: i for i, lbl in enumerate(SLEEP_LABELS)}
    sleep_inv_label  = {i: lbl for i, lbl in enumerate(SLEEP_LABELS)}

    # Feature importances
    stress_feat_imp = _map_importances(stress_pipeline, stress_feature_names)
    sleep_feat_imp  = _map_importances(sleep_pipeline,  sleep_feature_names)

    # Compute per-column medians and modes from training data.
    # predictor.py reads these as _STRESS_MEDIANS / _STRESS_MODES etc. with .get() fallbacks.
    stress_numeric = stress_X.select_dtypes(include="number")
    stress_object  = stress_X.select_dtypes(exclude="number")
    stress_X_medians = {col: float(stress_numeric[col].median()) for col in stress_numeric.columns}
    stress_X_modes   = {col: str(stress_object[col].mode().iloc[0])
                        for col in stress_object.columns if not stress_object[col].mode().empty}

    sleep_numeric = sleep_X.select_dtypes(include="number")
    sleep_object  = sleep_X.select_dtypes(exclude="number")
    sleep_X_medians = {col: float(sleep_numeric[col].median()) for col in sleep_numeric.columns}
    sleep_X_modes   = {col: str(sleep_object[col].mode().iloc[0])
                       for col in sleep_object.columns if not sleep_object[col].mode().empty}

    bundle = {
        # Stress model
        "stress_model":               stress_pipeline,
        "stress_feature_names":       stress_feature_names,
        "stress_label_map":           stress_label_map,
        "stress_inv_label":           stress_inv_label,
        "stress_feature_importances": stress_feat_imp,
        "stress_X_medians":           stress_X_medians,
        "stress_X_modes":             stress_X_modes,
        # Sleep model (classifier wrapper over regressor)
        "sleep_model":                sleep_classifier,
        "sleep_feature_names":        sleep_feature_names,
        "sleep_label_map":            sleep_label_map,
        "sleep_inv_label":            sleep_inv_label,
        "sleep_feature_importances":  sleep_feat_imp,
        "sleep_X_medians":            sleep_X_medians,
        "sleep_X_modes":              sleep_X_modes,
    }

    out = Path(__file__).parent / "model.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model bundle to {out}")
    print("Run `streamlit run app.py` to launch the UI.")


if __name__ == "__main__":
    main()
