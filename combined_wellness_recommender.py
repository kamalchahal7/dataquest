"""
Prototype combined wellness recommender.

This script combines:
1. The actionable stress classifier
2. A sleep quality classifier

High-level idea:
- Predict the user's stress class from actionable lifestyle inputs.
- Translate that stress prediction into a numeric stress level for the sleep model.
- Predict sleep quality using the sleep model.
- Search for small lifestyle changes that improve both stress and sleep.

Important note:
- Because the stress and sleep datasets come from different sources, this script
  uses a heuristic bridge between them.
- Specifically, predicted stress class from the stress model is mapped to the
  sleep model's numeric Stress Level input.
- This is a prototype recommender, not a medically validated optimizer.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


SLEEP_DATASET_PATH = Path("Sleep_health_and_lifestyle_dataset.csv")
STRESS_DATASET_PATH = Path("extended_stress_detection_data.csv")

SLEEP_TARGET_COLUMN = "Quality of Sleep"
SLEEP_CLASS_ORDER = ["Poor", "Moderate", "Good"]
SLEEP_CLASS_SCORE_ANCHORS = {
    "Poor": 4.5,
    "Moderate": 6.75,
    "Good": 8.75,
}
SLEEP_EXCLUDED_COLUMNS = [
    "Person ID",
    "Occupation",
    "Heart Rate",
    "Blood Pressure",
    "Sleep Disorder",
] + [SLEEP_TARGET_COLUMN]

STRESS_TARGET_COLUMN = "Stress_Detection"
STRESS_TARGET_ORDER = ["Low", "Medium", "High"]
STRESS_ACTIONABLE_COLUMNS = [
    "Sleep_Duration",
    "Sleep_Quality",
    "Wake_Up_Time",
    "Bed_Time",
    "Physical_Activity",
    "Screen_Time",
    "Caffeine_Intake",
    "Alcohol_Intake",
    "Smoking_Habit",
    "Work_Hours",
    "Travel_Time",
    "Social_Interactions",
    "Meditation_Practice",
    "Exercise_Type",
    "Marital_Status",
]

OUTPUT_ROOT = Path("outputs") / "combined_wellness"
RECOMMENDATION_OUTPUT_PATH = OUTPUT_ROOT / "recommendations" / "latest_combined_recommendation.csv"
SLEEP_STRESS_MEAN = 5.385026737967914
SLEEP_STRESS_CLASS_ANCHORS = {
    "Low": 3.2,
    "Medium": 5.3,
    "High": 7.8,
}
SLEEP_STRESS_CLASS_PROBABILITY_WEIGHTS = {
    "Low": 0.15,
    "Medium": 0.30,
    "High": 0.60,
}
STRESS_BRIDGE_STRENGTH = 0.85
MAX_RECOMMENDED_CHANGES = 2
MIN_GAIN_FOR_CAFFEINE_AND_ALCOHOL = 0.45
MIN_SLEEP_IMPROVEMENT_WHEN_STRESS_ALREADY_SATISFIED = 0.10

COMBINED_INPUT_FIELDS = [
    "Age",
    "Gender",
    "BMI Category",
    "Daily Steps",
    "Sleep Duration",
    "Physical Activity",
    "Screen Time",
    "Caffeine Intake",
    "Alcohol Intake",
    "Work Hours",
    "Travel Time",
    "Social Interactions",
    "Meditation Practice",
]


def ensure_output_directories() -> None:
    for folder in [OUTPUT_ROOT / "recommendations"]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_boolean_like_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    boolean_map = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
        "t": 1,
        "f": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
    }

    normalized = dataframe.copy()
    object_columns = normalized.select_dtypes(include=["object", "string"]).columns

    for column in object_columns:
        cleaned_strings = normalized[column].astype("string").str.strip().str.lower()
        non_missing_strings = cleaned_strings.dropna()

        if non_missing_strings.empty:
            continue

        unique_values = set(non_missing_strings.unique().tolist())
        if unique_values.issubset(set(boolean_map.keys())):
            normalized[column] = cleaned_strings.map(boolean_map)

    return normalized


def coerce_numeric_like_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    converted = dataframe.copy()

    for column in converted.columns:
        series = converted[column]
        if pd.api.types.is_numeric_dtype(series):
            continue

        numeric_version = pd.to_numeric(series, errors="coerce")
        non_missing_count = series.notna().sum()

        if non_missing_count == 0:
            continue

        convertible_ratio = numeric_version.notna().sum() / non_missing_count
        if convertible_ratio >= 0.80:
            converted[column] = numeric_version

    return converted


def split_sleep_blood_pressure(dataframe: pd.DataFrame) -> pd.DataFrame:
    transformed = dataframe.copy()
    if "Blood Pressure" in transformed.columns:
        transformed = transformed.drop(columns=["Blood Pressure"])
    return transformed


def sleep_score_to_label(score: float) -> str:
    if float(score) <= 6.0:
        return "Poor"
    if float(score) <= 7.5:
        return "Moderate"
    return "Good"


def engineer_stress_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    engineered = dataframe.copy()

    if "Wake_Up_Time" in engineered.columns:
        wake_dt = pd.to_datetime(engineered["Wake_Up_Time"], format="%I:%M %p", errors="coerce")
        engineered["Wake_Up_Hour"] = wake_dt.dt.hour + wake_dt.dt.minute / 60.0
        engineered = engineered.drop(columns=["Wake_Up_Time"])

    if "Bed_Time" in engineered.columns:
        bed_dt = pd.to_datetime(engineered["Bed_Time"], format="%I:%M %p", errors="coerce")
        bed_hour = bed_dt.dt.hour + bed_dt.dt.minute / 60.0
        engineered["Bed_Time_Hour"] = bed_hour.where(bed_hour >= 12, bed_hour + 24)
        engineered = engineered.drop(columns=["Bed_Time"])

    if {"Wake_Up_Hour", "Bed_Time_Hour"}.issubset(engineered.columns):
        engineered["Time_In_Bed_Hours"] = engineered["Wake_Up_Hour"] + 24 - engineered["Bed_Time_Hour"]

    return engineered


def clean_stress_negative_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    non_negative_columns = [
        "Sleep_Duration",
        "Sleep_Quality",
        "Physical_Activity",
        "Screen_Time",
        "Caffeine_Intake",
        "Alcohol_Intake",
        "Work_Hours",
        "Travel_Time",
        "Social_Interactions",
    ]

    for column in non_negative_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned.loc[cleaned[column] < 0, column] = np.nan

    return cleaned


def prepare_sleep_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(SLEEP_DATASET_PATH)
    predictor_columns = [column for column in df.columns if column not in SLEEP_EXCLUDED_COLUMNS]
    df = df[predictor_columns + [SLEEP_TARGET_COLUMN]].copy()
    df = df.dropna(subset=[SLEEP_TARGET_COLUMN]).copy()
    df = split_sleep_blood_pressure(df)
    predictor_columns = [column for column in df.columns if column != SLEEP_TARGET_COLUMN]
    df[predictor_columns] = normalize_boolean_like_columns(df[predictor_columns])
    df[predictor_columns] = coerce_numeric_like_columns(df[predictor_columns])
    sleep_scores = pd.to_numeric(df[SLEEP_TARGET_COLUMN], errors="coerce")
    sleep_labels = sleep_scores.map(sleep_score_to_label)
    y = pd.Categorical(sleep_labels, categories=SLEEP_CLASS_ORDER, ordered=True)
    return df[predictor_columns].copy(), pd.Series(y.astype(str), index=df.index, name="Sleep_Class")


def prepare_stress_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(STRESS_DATASET_PATH)
    df = df[STRESS_ACTIONABLE_COLUMNS + [STRESS_TARGET_COLUMN]].copy()
    df = clean_stress_negative_values(df)
    df = engineer_stress_time_features(df)
    predictor_columns = [column for column in df.columns if column != STRESS_TARGET_COLUMN]
    df[predictor_columns] = normalize_boolean_like_columns(df[predictor_columns])
    df[predictor_columns] = coerce_numeric_like_columns(df[predictor_columns])
    df = df[df[STRESS_TARGET_COLUMN].isin(STRESS_TARGET_ORDER)].copy()
    y = pd.Categorical(df[STRESS_TARGET_COLUMN], categories=STRESS_TARGET_ORDER, ordered=True)
    return df[predictor_columns].copy(), pd.Series(y.astype(str), index=df.index, name=STRESS_TARGET_COLUMN)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_sleep_model(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=2,
                    subsample=0.7,
                ),
            ),
        ]
    )


def build_stress_model(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features="sqrt",
                ),
            ),
        ]
    )


def prompt_with_default(prompt_text: str, default_value: object) -> str:
    response = input(f"{prompt_text} [{default_value}]: ").strip()
    return response if response else str(default_value)


def get_default_value(series: pd.Series) -> object:
    if pd.api.types.is_numeric_dtype(series):
        return round(float(series.median()), 2)
    return series.mode(dropna=True).iloc[0]


def build_combined_user_profiles(
    stress_X: pd.DataFrame,
    sleep_X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ask for one reduced set of inputs and map them into both models.
    """
    print("\nEnter a smaller set of daily-life inputs. Press Enter to accept the default.")

    combined_defaults = {
        "Age": get_default_value(sleep_X["Age"]),
        "Gender": get_default_value(sleep_X["Gender"]),
        "BMI Category": get_default_value(sleep_X["BMI Category"]),
        "Daily Steps": get_default_value(sleep_X["Daily Steps"]),
        "Sleep Duration": get_default_value(stress_X["Sleep_Duration"]),
        "Physical Activity": get_default_value(stress_X["Physical_Activity"]),
        "Screen Time": get_default_value(stress_X["Screen_Time"]),
        "Caffeine Intake": get_default_value(stress_X["Caffeine_Intake"]),
        "Alcohol Intake": get_default_value(stress_X["Alcohol_Intake"]),
        "Work Hours": get_default_value(stress_X["Work_Hours"]),
        "Travel Time": get_default_value(stress_X["Travel_Time"]),
        "Social Interactions": get_default_value(stress_X["Social_Interactions"]),
        "Meditation Practice": "Yes" if float(get_default_value(stress_X["Meditation_Practice"])) >= 0.5 else "No",
    }

    combined_profile = {}
    numeric_combined_fields = {
        "Age",
        "Daily Steps",
        "Sleep Duration",
        "Physical Activity",
        "Screen Time",
        "Caffeine Intake",
        "Alcohol Intake",
        "Work Hours",
        "Travel Time",
        "Social Interactions",
    }

    for field in COMBINED_INPUT_FIELDS:
        user_value = prompt_with_default(field, combined_defaults[field])
        if field in numeric_combined_fields:
            combined_profile[field] = pd.to_numeric(user_value, errors="coerce")
        else:
            combined_profile[field] = user_value

    stress_profile = pd.DataFrame([{
        "Sleep_Duration": combined_profile["Sleep Duration"],
        "Sleep_Quality": get_default_value(stress_X["Sleep_Quality"]),
        "Physical_Activity": combined_profile["Physical Activity"],
        "Screen_Time": combined_profile["Screen Time"],
        "Caffeine_Intake": combined_profile["Caffeine Intake"],
        "Alcohol_Intake": combined_profile["Alcohol Intake"],
        "Smoking_Habit": get_default_value(stress_X["Smoking_Habit"]),
        "Work_Hours": combined_profile["Work Hours"],
        "Travel_Time": combined_profile["Travel Time"],
        "Social_Interactions": combined_profile["Social Interactions"],
        "Meditation_Practice": 1.0 if str(combined_profile["Meditation Practice"]).strip().lower() in {"yes", "y", "1", "true"} else 0.0,
        "Exercise_Type": get_default_value(stress_X["Exercise_Type"]),
        "Marital_Status": get_default_value(stress_X["Marital_Status"]),
        "Wake_Up_Hour": 7.0,
        "Bed_Time_Hour": 31.0 - float(combined_profile["Sleep Duration"]),
        "Time_In_Bed_Hours": float(combined_profile["Sleep Duration"]) + 1.0,
    }], columns=stress_X.columns)

    sleep_profile = pd.DataFrame([{
        "Gender": combined_profile["Gender"],
        "Age": combined_profile["Age"],
        "Sleep Duration": combined_profile["Sleep Duration"],
        "Physical Activity Level": float(combined_profile["Physical Activity"]) * 20.0,
        "BMI Category": combined_profile["BMI Category"],
        "Daily Steps": combined_profile["Daily Steps"],
        "Stress Level": np.nan,
    }], columns=sleep_X.columns)

    return stress_profile, sleep_profile


def stress_probabilities_to_sleep_stress_level(stress_probabilities: dict[str, float]) -> float:
    """
    Calibrated bridge between the stress classifier and sleep classifier.

    We first map class probabilities onto empirical stress anchors from the
    sleep dataset, then shrink the result back toward the dataset mean.
    This keeps small classification shifts from causing unrealistic sleep jumps.
    """
    # Reweight the classifier probabilities before bridging them into the sleep model.
    adjusted_weighted_sum = 0.0
    adjusted_probability_sum = 0.0

    for label, stress_value in SLEEP_STRESS_CLASS_ANCHORS.items():
        probability = float(stress_probabilities.get(label, 0.0))
        bridge_weight = float(SLEEP_STRESS_CLASS_PROBABILITY_WEIGHTS.get(label, 1.0))
        adjusted_probability = probability * bridge_weight
        adjusted_weighted_sum += adjusted_probability * stress_value
        adjusted_probability_sum += adjusted_probability

    if adjusted_probability_sum <= 0:
        return SLEEP_STRESS_MEAN

    raw_bridge_value = adjusted_weighted_sum / adjusted_probability_sum
    damped_bridge_value = SLEEP_STRESS_MEAN + STRESS_BRIDGE_STRENGTH * (
        raw_bridge_value - SLEEP_STRESS_MEAN
    )

    return float(min(8.0, max(3.0, damped_bridge_value)))


def default_target_stress_class(current_stress_class: str) -> str:
    if current_stress_class == "High":
        return "Medium"
    if current_stress_class == "Medium":
        return "Medium"
    return "Medium"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stress_wellness_probability_map(profile_df: pd.DataFrame, classes: list[str]) -> dict[str, float]:
    row = profile_df.iloc[0]
    sleep_duration = float(row.get("Sleep_Duration", 6.5))
    physical_activity = float(row.get("Physical_Activity", 30.0))
    screen_time = float(row.get("Screen_Time", 4.0))
    caffeine = float(row.get("Caffeine_Intake", 2.0))
    alcohol = float(row.get("Alcohol_Intake", 0.0))
    work_hours = float(row.get("Work_Hours", 8.0))
    social = float(row.get("Social_Interactions", 2.0))
    smoking = float(row.get("Smoking_Habit", 0.0))
    meditation = float(row.get("Meditation_Practice", 0.0))

    sleep_score = _clip01(1.0 - min(abs(sleep_duration - 7.5) / 3.0, 1.0))
    activity_score = _clip01(physical_activity / 45.0)
    screen_score = _clip01(1.0 - (screen_time / 8.0))
    caffeine_score = _clip01(1.0 - (caffeine / 6.0))
    alcohol_score = _clip01(1.0 - (alcohol / 4.0))
    work_score = _clip01(1.0 - (max(0.0, work_hours - 8.0) / 5.0))
    social_score = _clip01(social / 4.0)
    smoking_score = 1.0 if smoking <= 0.0 else 0.0
    meditation_score = 1.0 if meditation > 0.0 else 0.35

    wellness = _clip01(
        (0.22 * sleep_score)
        + (0.13 * activity_score)
        + (0.13 * screen_score)
        + (0.09 * caffeine_score)
        + (0.07 * alcohol_score)
        + (0.12 * work_score)
        + (0.10 * social_score)
        + (0.08 * smoking_score)
        + (0.06 * meditation_score)
    )

    high = _clip01((0.48 - wellness) / 0.30)
    low = _clip01((wellness - 0.58) / 0.25)
    medium = max(0.0, 1.0 - high - low)
    base = {"High": high, "Low": low, "Medium": medium}
    total = sum(base.values()) or 1.0
    return {label: base.get(label, 0.0) / total for label in classes}


def predict_stress(model: Pipeline, profile_df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    raw_probabilities = model.predict_proba(profile_df)[0]
    classes = list(model.named_steps["classifier"].classes_)
    raw_map = {
        label: float(probability)
        for label, probability in zip(classes, raw_probabilities)
    }
    wellness_map = _stress_wellness_probability_map(profile_df, classes)
    probability_map = {
        label: (0.55 * raw_map.get(label, 0.0)) + (0.45 * wellness_map.get(label, 0.0))
        for label in classes
    }
    total = sum(probability_map.values()) or 1.0
    probability_map = {label: value / total for label, value in probability_map.items()}

    high = probability_map.get("High", 0.0)
    medium = probability_map.get("Medium", 0.0)
    low = probability_map.get("Low", 0.0)
    if high >= 0.60 and (high - medium) >= 0.12:
        predicted_class = "High"
    elif low >= 0.46 and (low - medium) >= 0.08:
        predicted_class = "Low"
    else:
        predicted_class = "Medium"
    return predicted_class, probability_map


def predict_sleep_class(model: Pipeline, profile_df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    predicted_class = model.predict(profile_df)[0]
    probabilities = model.predict_proba(profile_df)[0]
    probability_map = {
        label: float(probability)
        for label, probability in zip(model.named_steps["classifier"].classes_, probabilities)
    }
    return str(predicted_class), probability_map


def sleep_probabilities_to_score(sleep_probabilities: dict[str, float]) -> float:
    return float(
        sum(
            float(sleep_probabilities.get(label, 0.0)) * SLEEP_CLASS_SCORE_ANCHORS[label]
            for label in SLEEP_CLASS_ORDER
        )
    )


def _sleep_score_adjustment(profile_df: pd.DataFrame) -> float:
    row = profile_df.iloc[0]
    sleep_duration = float(row.get("Sleep Duration", 6.5))
    activity = float(row.get("Physical Activity Level", 30.0))
    daily_steps = float(row.get("Daily Steps", 7000.0))
    stress_level = float(row.get("Stress Level", SLEEP_STRESS_MEAN))

    duration_bonus = max(0.0, min(1.0, (sleep_duration - 6.5) / 1.5)) * 0.7
    activity_bonus = max(0.0, min(1.0, (activity - 30.0) / 40.0)) * 0.35
    steps_bonus = max(0.0, min(1.0, (daily_steps - 7000.0) / 2500.0)) * 0.3
    stress_bonus = max(0.0, min(1.0, (5.4 - stress_level) / 2.2)) * 0.95
    return float(duration_bonus + activity_bonus + steps_bonus + stress_bonus)


def predict_sleep_quality(model: Pipeline, profile_df: pd.DataFrame) -> float:
    _, probability_map = predict_sleep_class(model, profile_df)
    base_score = sleep_probabilities_to_score(probability_map)
    adjusted_score = base_score + _sleep_score_adjustment(profile_df)
    return float(min(9.4, max(1.0, adjusted_score)))


def update_sleep_profile_from_stress_and_actions(
    sleep_profile: pd.DataFrame,
    predicted_stress_probabilities: dict[str, float],
    activity_points_increase: float,
    daily_steps_increase: float,
) -> pd.DataFrame:
    updated = sleep_profile.copy()
    updated.at[updated.index[0], "Stress Level"] = stress_probabilities_to_sleep_stress_level(
        predicted_stress_probabilities
    )

    if "Physical Activity Level" in updated.columns:
        updated.at[updated.index[0], "Physical Activity Level"] = (
            float(updated.iloc[0]["Physical Activity Level"]) + activity_points_increase
        )

    if "Daily Steps" in updated.columns:
        updated.at[updated.index[0], "Daily Steps"] = (
            float(updated.iloc[0]["Daily Steps"]) + daily_steps_increase
        )

    return updated


def generate_combined_candidates(
    stress_profile: pd.DataFrame,
) -> list[dict[str, object]]:
    """
    Small bounded search over realistic, user-actionable changes.

    Assumptions for this prototype:
    - physical activity can increase by up to 2 units in the stress dataset
    - screen time can decrease by up to 2 hours
    - caffeine can decrease by up to 2 drinks
    - social interactions can increase by up to 2 units
    - meditation can switch from No to Yes
    - daily steps can increase up to 3000
    - sleep-model physical activity level gets a linked increase from activity change
    """
    current = stress_profile.iloc[0]

    candidates = []
    for activity_delta in [0, 1, 2, 3]:
        for screen_delta in [0, 1, 2, 3]:
            for caffeine_delta in [0, 1, 2]:
                for social_delta in [0, 1, 2]:
                    for alcohol_delta in [0, 1]:
                        for meditation_target in ["keep", "Yes"]:
                            for steps_delta in [0, 1000, 2000, 3000]:
                                candidate = {
                                    "activity_delta": activity_delta,
                                    "screen_delta": screen_delta,
                                    "caffeine_delta": caffeine_delta,
                                    "social_delta": social_delta,
                                    "alcohol_delta": alcohol_delta,
                                    "meditation_target": meditation_target,
                                    "steps_delta": steps_delta,
                                }
                                _, changed_factor_count = candidate_score(candidate, stress_profile)
                                if 0 < changed_factor_count <= MAX_RECOMMENDED_CHANGES:
                                    candidates.append(candidate)

    return candidates


def rank_candidates_by_change_size(
    candidates: list[dict[str, object]],
    original_stress_profile: pd.DataFrame,
) -> list[dict[str, object]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate_score(candidate, original_stress_profile)[0],
            candidate_score(candidate, original_stress_profile)[1],
        ),
    )


def apply_stress_candidate(stress_profile: pd.DataFrame, candidate: dict[str, object]) -> pd.DataFrame:
    updated = stress_profile.copy()

    updated.at[updated.index[0], "Physical_Activity"] = max(
        0.0,
        float(updated.iloc[0]["Physical_Activity"]) + float(candidate["activity_delta"]),
    )
    updated.at[updated.index[0], "Screen_Time"] = max(
        0.0,
        float(updated.iloc[0]["Screen_Time"]) - float(candidate["screen_delta"]),
    )
    updated.at[updated.index[0], "Caffeine_Intake"] = max(
        0.0,
        float(updated.iloc[0]["Caffeine_Intake"]) - float(candidate["caffeine_delta"]),
    )
    updated.at[updated.index[0], "Alcohol_Intake"] = max(
        0.0,
        float(updated.iloc[0]["Alcohol_Intake"]) - float(candidate["alcohol_delta"]),
    )
    updated.at[updated.index[0], "Social_Interactions"] = max(
        0.0,
        float(updated.iloc[0]["Social_Interactions"]) + float(candidate["social_delta"]),
    )

    if candidate["meditation_target"] == "Yes":
        updated.at[updated.index[0], "Meditation_Practice"] = 1.0

    return updated


def candidate_score(
    candidate: dict[str, object],
    original_stress_profile: pd.DataFrame,
) -> tuple[float, int]:
    original_meditation = float(original_stress_profile.iloc[0]["Meditation_Practice"])
    meditation_changes = (
        candidate["meditation_target"] == "Yes" and original_meditation < 0.5
    )

    total_change = (
        float(candidate["activity_delta"])
        + float(candidate["screen_delta"])
        + float(candidate["caffeine_delta"])
        + float(candidate["alcohol_delta"])
        + float(candidate["social_delta"])
        + (1.0 if meditation_changes else 0.0)
        + float(candidate["steps_delta"]) / 1000.0
    )
    changed_factor_count = sum(
        [
            candidate["activity_delta"] > 0,
            candidate["screen_delta"] > 0,
            candidate["caffeine_delta"] > 0,
            candidate["alcohol_delta"] > 0,
            candidate["social_delta"] > 0,
            meditation_changes,
            candidate["steps_delta"] > 0,
        ]
    )
    return total_change, changed_factor_count


def candidate_effort_score(
    candidate: dict[str, object],
    original_stress_profile: pd.DataFrame,
) -> float:
    row = original_stress_profile.iloc[0]
    original_meditation = float(row["Meditation_Practice"])
    meditation_changes = candidate["meditation_target"] == "Yes" and original_meditation < 0.5

    effort = 0.0
    effort += float(candidate["activity_delta"]) / max(1.0, float(row["Physical_Activity"]) * 0.5, 1.0)
    effort += float(candidate["screen_delta"]) / max(1.0, float(row["Screen_Time"]) * 0.5, 1.0)
    effort += float(candidate["caffeine_delta"]) / max(1.0, float(row["Caffeine_Intake"]) * 0.5, 1.0)
    effort += float(candidate["alcohol_delta"]) / max(1.0, float(row["Alcohol_Intake"]) * 0.5, 1.0)
    effort += float(candidate["social_delta"]) / max(1.0, float(row["Social_Interactions"]) * 0.5, 1.0)
    effort += float(candidate["steps_delta"]) / max(1000.0, float(max(1.0, row.get("Daily Steps", 7000.0))) * 0.5)
    if meditation_changes:
        effort += 0.6

    _, changed_factor_count = candidate_score(candidate, original_stress_profile)
    return float(effort + max(0, changed_factor_count - 1) * 0.35)


def candidate_power_score(
    candidate: dict[str, object],
    original_stress_profile: pd.DataFrame,
    current_sleep_quality: float,
    predicted_sleep_quality: float,
    current_high_probability: float,
    predicted_probabilities: dict[str, float],
    current_target_or_better_probability: float,
    target_rank: int,
) -> float:
    sleep_gain = max(0.0, predicted_sleep_quality - current_sleep_quality)
    high_probability_reduction = max(0.0, current_high_probability - predicted_probabilities.get("High", 0.0))
    target_or_better_probability_gain = max(
        0.0,
        sum(predicted_probabilities.get(label, 0.0) for label in STRESS_TARGET_ORDER[: target_rank + 1])
        - current_target_or_better_probability,
    )
    benefit = (
        1.8 * sleep_gain
        + 2.2 * target_or_better_probability_gain
        + 1.4 * high_probability_reduction
    )
    effort = candidate_effort_score(candidate, original_stress_profile)
    return float(benefit / max(0.25, effort))


def allow_candidate_under_policy(
    candidate: dict[str, object],
    changed_factor_count: int,
    predicted_sleep_quality: float,
    current_sleep_quality: float,
    predicted_rank: int,
    target_rank: int,
    stress_target_already_satisfied: bool,
) -> bool:
    """
    Enforce practical recommendation rules.
    """
    if changed_factor_count > MAX_RECOMMENDED_CHANGES:
        return False

    reducing_caffeine = candidate["caffeine_delta"] > 0
    reducing_alcohol = candidate["alcohol_delta"] > 0
    if reducing_caffeine and reducing_alcohol:
        if predicted_sleep_quality - current_sleep_quality < MIN_GAIN_FOR_CAFFEINE_AND_ALCOHOL:
            return False

    if stress_target_already_satisfied:
        if predicted_rank > target_rank:
            return False

        # When stress is already good enough, only surface small changes that
        # produce a meaningful sleep-quality gain.
        if predicted_sleep_quality - current_sleep_quality < MIN_SLEEP_IMPROVEMENT_WHEN_STRESS_ALREADY_SATISFIED:
            return False

    return True


def allow_candidate_for_near_miss_sleep_fallback(
    candidate: dict[str, object],
    changed_factor_count: int,
    predicted_sleep_quality: float,
    current_sleep_quality: float,
    predicted_rank: int,
    target_rank: int,
) -> bool:
    """
    Looser fallback policy used only when no stronger recommendation exists.
    Keeps the practical limits, but allows smaller sleep gains.
    """
    if changed_factor_count > MAX_RECOMMENDED_CHANGES:
        return False

    if predicted_rank > target_rank:
        return False

    reducing_caffeine = candidate["caffeine_delta"] > 0
    reducing_alcohol = candidate["alcohol_delta"] > 0
    if reducing_caffeine and reducing_alcohol:
        if predicted_sleep_quality - current_sleep_quality < MIN_GAIN_FOR_CAFFEINE_AND_ALCOHOL:
            return False

    return predicted_sleep_quality > current_sleep_quality


def evaluate_sleep_model_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Cross-validation evaluation for the sleep quality classification model.
    Uses 5-fold stratified cross-validation with classification metrics.
    """
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
    }
    
    cv_results = cross_validate(
        model, X, y, cv=cv5, scoring=scoring_metrics, return_train_score=True
    )
    
    y_pred = cross_val_predict(model, X, y, cv=cv5)
    
    results = {
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision_mean': cv_results['test_precision_macro'].mean(),
        'precision_std': cv_results['test_precision_macro'].std(),
        'recall_mean': cv_results['test_recall_macro'].mean(),
        'recall_std': cv_results['test_recall_macro'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'f1_weighted_mean': cv_results['test_f1_weighted'].mean(),
        'f1_weighted_std': cv_results['test_f1_weighted'].std(),
        'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        'test_predictions': y_pred,
    }
    
    return results


def evaluate_stress_model_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Cross-validation evaluation for the stress classification model.
    Uses 5-fold stratified cross-validation with classification metrics.
    """
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
    }
    
    cv_results = cross_validate(
        model, X, y, cv=cv5, scoring=scoring_metrics, return_train_score=True
    )
    
    # Get predictions for additional metrics
    y_pred = cross_val_predict(model, X, y, cv=cv5)
    
    results = {
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision_mean': cv_results['test_precision_macro'].mean(),
        'precision_std': cv_results['test_precision_macro'].std(),
        'recall_mean': cv_results['test_recall_macro'].mean(),
        'recall_std': cv_results['test_recall_macro'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'f1_weighted_mean': cv_results['test_f1_weighted'].mean(),
        'f1_weighted_std': cv_results['test_f1_weighted'].std(),
        'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        'test_predictions': y_pred,
    }
    
    return results


def print_sleep_cv_report(results: dict, y_true: pd.Series) -> None:
    """Print formatted cross-validation report for sleep model."""
    print(f"\n{'─' * 70}")
    print(f"SLEEP QUALITY MODEL — 5-Fold Cross-Validation Results")
    print(f"{'─' * 70}")
    print(f"Accuracy        : {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Precision       : {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall          : {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    print(f"F1 (Macro)      : {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1 (Weighted)   : {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    print(f"Train Accuracy  : {results['train_accuracy_mean']:.4f} (overfitting check)")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, results['test_predictions'], labels=SLEEP_CLASS_ORDER))

    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, results['test_predictions']))


def print_stress_cv_report(results: dict, y_true: pd.Series) -> None:
    """Print formatted cross-validation report for stress model."""
    print(f"\n{'─' * 70}")
    print(f"STRESS LEVEL MODEL — 5-Fold Cross-Validation Results")
    print(f"{'─' * 70}")
    print(f"Accuracy        : {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Precision       : {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall          : {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    print(f"F1 (Macro)      : {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1 (Weighted)   : {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    print(f"Train Accuracy  : {results['train_accuracy_mean']:.4f} (overfitting check)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, results['test_predictions'])
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, results['test_predictions']))


def main() -> None:
    ensure_output_directories()

    if not SLEEP_DATASET_PATH.exists() or not STRESS_DATASET_PATH.exists():
        print("Error: One or both required datasets were not found in the data folder.")
        return

    print("Preparing datasets and training both models...")

    sleep_X, sleep_y = prepare_sleep_data()
    stress_X, stress_y = prepare_stress_data()

    sleep_X_train, _, sleep_y_train, _ = train_test_split(
        sleep_X,
        sleep_y,
        test_size=0.20,
        random_state=42,
    )
    stress_X_train, _, stress_y_train, _ = train_test_split(
        stress_X,
        stress_y,
        test_size=0.20,
        random_state=42,
        stratify=stress_y,
    )

    sleep_model = build_sleep_model(sleep_X)
    stress_model = build_stress_model(stress_X)
    sleep_model.fit(sleep_X_train, sleep_y_train)
    stress_model.fit(stress_X_train, stress_y_train)

    # ── Cross-Validation Evaluation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION EVALUATION RESULTS")
    print("=" * 70)
    
    # Evaluate sleep model with 5-fold cross-validation
    print("\n🧪 Evaluating Sleep Quality Model...")
    sleep_cv_results = evaluate_sleep_model_cv(sleep_model, sleep_X, sleep_y)
    print_sleep_cv_report(sleep_cv_results, sleep_y)
    
    # Evaluate stress model with 5-fold cross-validation
    print("\n🧪 Evaluating Stress Level Model...")
    stress_cv_results = evaluate_stress_model_cv(stress_model, stress_X, stress_y)
    print_stress_cv_report(stress_cv_results, stress_y)
    
    print("\n" + "=" * 70)
    print("✅ Model Evaluation Complete")
    print("=" * 70)

    stress_profile, sleep_profile = build_combined_user_profiles(stress_X, sleep_X)

    current_stress_class, current_stress_probs = predict_stress(stress_model, stress_profile)
    current_implied_stress_level = stress_probabilities_to_sleep_stress_level(
        current_stress_probs
    )
    current_sleep_profile = update_sleep_profile_from_stress_and_actions(
        sleep_profile=sleep_profile,
        predicted_stress_probabilities=current_stress_probs,
        activity_points_increase=0.0,
        daily_steps_increase=0.0,
    )
    current_sleep_quality = predict_sleep_quality(sleep_model, current_sleep_profile)

    print("\nCurrent predicted wellness state:")
    print(f"  Predicted stress class       : {current_stress_class}")
    print(f"  Stress probabilities         : {current_stress_probs}")
    print(f"  Implied sleep stress level   : {current_implied_stress_level:.2f}")
    print(f"  Predicted sleep quality      : {current_sleep_quality:.2f}")

    target_sleep_quality_raw = prompt_with_default(
        "Target sleep quality",
        round(current_sleep_quality + 0.5, 2),
    )
    target_sleep_quality = float(pd.to_numeric(target_sleep_quality_raw, errors="coerce"))
    effective_target_sleep_quality = max(target_sleep_quality, current_sleep_quality)
    target_stress_class = prompt_with_default(
        "Target stress class (Low / Medium / High)",
        default_target_stress_class(current_stress_class),
    ).title()

    if target_stress_class not in STRESS_TARGET_ORDER:
        print("Error: Stress target must be Low, Medium, or High.")
        return

    best_result = None
    closest_result = None
    best_improvement_result = None
    best_sleep_near_miss_result = None
    target_rank = STRESS_TARGET_ORDER.index(target_stress_class)
    current_high_probability = current_stress_probs.get("High", 0.0)
    current_rank = STRESS_TARGET_ORDER.index(current_stress_class)
    stress_target_already_satisfied = current_rank <= target_rank
    current_target_or_better_probability = sum(
        current_stress_probs.get(label, 0.0)
        for label in STRESS_TARGET_ORDER[: target_rank + 1]
    )

    ranked_candidates = rank_candidates_by_change_size(
        generate_combined_candidates(stress_profile),
        original_stress_profile=stress_profile,
    )

    for candidate in ranked_candidates:
        candidate_stress_profile = apply_stress_candidate(stress_profile, candidate)
        predicted_stress_class, stress_probs = predict_stress(stress_model, candidate_stress_profile)
        predicted_rank = STRESS_TARGET_ORDER.index(predicted_stress_class)

        candidate_sleep_profile = update_sleep_profile_from_stress_and_actions(
            sleep_profile=sleep_profile,
            predicted_stress_probabilities=stress_probs,
            activity_points_increase=float(candidate["activity_delta"]) * 10.0,
            daily_steps_increase=float(candidate["steps_delta"]),
        )
        predicted_sleep_quality = predict_sleep_quality(sleep_model, candidate_sleep_profile)

        change_score, changed_factor_count = candidate_score(
            candidate,
            original_stress_profile=stress_profile,
        )

        result = {
            "candidate": candidate,
            "predicted_stress_class": predicted_stress_class,
            "stress_probabilities": stress_probs,
            "predicted_sleep_quality": predicted_sleep_quality,
            "change_score": change_score,
            "changed_factor_count": changed_factor_count,
        }

        if not allow_candidate_under_policy(
            candidate=candidate,
            changed_factor_count=changed_factor_count,
            predicted_sleep_quality=predicted_sleep_quality,
            current_sleep_quality=current_sleep_quality,
            predicted_rank=predicted_rank,
            target_rank=target_rank,
            stress_target_already_satisfied=stress_target_already_satisfied,
        ):
            continue

        if changed_factor_count == 0:
            continue

        result["high_probability_reduction"] = current_high_probability - stress_probs.get("High", 0.0)
        result["target_or_better_probability_gain"] = (
            sum(stress_probs.get(label, 0.0) for label in STRESS_TARGET_ORDER[: target_rank + 1])
            - current_target_or_better_probability
        )

        if allow_candidate_for_near_miss_sleep_fallback(
            candidate=candidate,
            changed_factor_count=changed_factor_count,
            predicted_sleep_quality=predicted_sleep_quality,
            current_sleep_quality=current_sleep_quality,
            predicted_rank=predicted_rank,
            target_rank=target_rank,
        ):
            near_miss_key = (
                -round(result["predicted_sleep_quality"] - current_sleep_quality, 6),
                result["changed_factor_count"],
                result["change_score"],
                -round(result["target_or_better_probability_gain"], 6),
                -round(result["high_probability_reduction"], 6),
            )
            if (
                best_sleep_near_miss_result is None
                or near_miss_key < best_sleep_near_miss_result["near_miss_key"]
            ):
                result["near_miss_key"] = near_miss_key
                best_sleep_near_miss_result = result

        sleep_gap = max(0.0, effective_target_sleep_quality - predicted_sleep_quality)
        stress_gap = max(0, predicted_rank - target_rank)
        closeness_key = (
            stress_gap,
            round(sleep_gap, 4),
            -round(result["target_or_better_probability_gain"], 6),
            -round(result["high_probability_reduction"], 6),
            result["change_score"],
            result["changed_factor_count"],
        )

        if closest_result is None or closeness_key < closest_result["closeness_key"]:
            result["closeness_key"] = closeness_key
            closest_result = result

        if stress_target_already_satisfied:
            improvement_key = (
                -round(result["predicted_sleep_quality"] - current_sleep_quality, 6),
                result["changed_factor_count"],
                result["change_score"],
                -round(result["target_or_better_probability_gain"], 6),
                -round(result["high_probability_reduction"], 6),
            )
            qualifies_as_improvement = (
                result["changed_factor_count"] > 0
                and predicted_rank <= target_rank
                and (
                    result["predicted_sleep_quality"] - current_sleep_quality
                    >= MIN_SLEEP_IMPROVEMENT_WHEN_STRESS_ALREADY_SATISFIED
                )
            )
        else:
            improvement_key = (
                -round(result["target_or_better_probability_gain"], 6),
                -round(result["high_probability_reduction"], 6),
                -round(result["predicted_sleep_quality"], 6),
                result["change_score"],
                result["changed_factor_count"],
            )
            qualifies_as_improvement = (
                result["changed_factor_count"] > 0
                and (
                    result["target_or_better_probability_gain"] > 0
                    or result["high_probability_reduction"] > 0
                    or result["predicted_sleep_quality"] > current_sleep_quality
                )
            )

        if qualifies_as_improvement:
            if best_improvement_result is None or improvement_key < best_improvement_result["improvement_key"]:
                result["improvement_key"] = improvement_key
                best_improvement_result = result

        if predicted_rank > target_rank:
            continue

        if predicted_sleep_quality < effective_target_sleep_quality:
            continue

        if best_result is None:
            best_result = result
            continue

        if stress_target_already_satisfied:
            current_key = (
                round(max(0.0, effective_target_sleep_quality - result["predicted_sleep_quality"]), 6),
                result["changed_factor_count"],
                result["change_score"],
                -result["predicted_sleep_quality"],
                -result["target_or_better_probability_gain"],
            )
            best_key = (
                round(max(0.0, effective_target_sleep_quality - best_result["predicted_sleep_quality"]), 6),
                best_result["changed_factor_count"],
                best_result["change_score"],
                -best_result["predicted_sleep_quality"],
                -best_result["target_or_better_probability_gain"],
            )
        else:
            current_key = (
                STRESS_TARGET_ORDER.index(result["predicted_stress_class"]),
                result["changed_factor_count"],
                result["change_score"],
                -result["predicted_sleep_quality"],
            )
            best_key = (
                STRESS_TARGET_ORDER.index(best_result["predicted_stress_class"]),
                best_result["changed_factor_count"],
                best_result["change_score"],
                -best_result["predicted_sleep_quality"],
            )

        if current_key < best_key:
            best_result = result

    if best_result is not None:
        result_to_show = best_result
    elif best_improvement_result is not None:
        result_to_show = best_improvement_result
    elif best_sleep_near_miss_result is not None:
        result_to_show = best_sleep_near_miss_result
    else:
        result_to_show = closest_result

    if result_to_show is None:
        print("\nNo worthwhile recommendation was found under the current policy limits.")
        print("Try a lower sleep-quality target, a less aggressive stress target, or allow more changes.")

        recommendation_df = pd.DataFrame(
            [
                {
                    "current_stress_class": current_stress_class,
                    "current_implied_sleep_stress_level": current_implied_stress_level,
                    "current_sleep_duration_input": float(stress_profile.iloc[0]["Sleep_Duration"]),
                    "current_sleep_quality_prediction": current_sleep_quality,
                    "target_stress_class": target_stress_class,
                    "target_sleep_quality": target_sleep_quality,
                    "effective_sleep_quality_floor": effective_target_sleep_quality,
                    "exact_goal_hit": False,
                    "recommendation_found": False,
                    "recommended_activity_delta": 0,
                    "recommended_screen_time_reduction": 0,
                    "recommended_caffeine_reduction": 0,
                    "recommended_alcohol_reduction": 0,
                    "recommended_social_interaction_increase": 0,
                    "recommended_meditation_target": "keep",
                    "recommended_daily_steps_increase": 0,
                    "predicted_stress_class_after_change": current_stress_class,
                    "predicted_implied_sleep_stress_level_after_change": current_implied_stress_level,
                    "predicted_sleep_quality_after_change": current_sleep_quality,
                    "high_stress_probability_reduction": 0.0,
                    "target_or_better_probability_gain": 0.0,
                }
            ]
        )
        recommendation_df.to_csv(RECOMMENDATION_OUTPUT_PATH, index=False)
        print(f"\nSaved combined recommendation to: {RECOMMENDATION_OUTPUT_PATH.resolve()}")
        return

    candidate = result_to_show["candidate"]

    if best_result is None and result_to_show is best_sleep_near_miss_result:
        print("\nNo exact recommendation hit both goals, so here is the best near-miss sleep recommendation.")
        print(f"  Predicted stress class        : {result_to_show['predicted_stress_class']}")
        print(f"  Predicted sleep quality       : {result_to_show['predicted_sleep_quality']:.2f}")
        print(
            "  Sleep quality change         : "
            f"{result_to_show['predicted_sleep_quality'] - current_sleep_quality:+.2f}"
        )
    elif best_result is None:
        print("\nNo exact recommendation hit both goals, so here is the best improvement found.")
        print(f"  Closest predicted stress class : {result_to_show['predicted_stress_class']}")
        print(f"  Closest predicted sleep quality: {result_to_show['predicted_sleep_quality']:.2f}")
        print(f"  High-stress probability change : {result_to_show['high_probability_reduction']:+.3f}")
    else:
        print("\nRecommended combined changes:")

    print(f"  Increase physical activity by : {candidate['activity_delta']}")
    print(f"  Reduce screen time by         : {candidate['screen_delta']} hours")
    print(f"  Reduce caffeine by            : {candidate['caffeine_delta']} drinks")
    print(f"  Reduce alcohol by             : {candidate['alcohol_delta']} drinks")
    print(f"  Increase social interaction by: {candidate['social_delta']}")
    print(f"  Meditation target             : {candidate['meditation_target']}")
    print(f"  Increase daily steps by       : {candidate['steps_delta']}")
    print(f"  Predicted stress class        : {result_to_show['predicted_stress_class']}")
    print(f"  Implied sleep stress level    : {stress_probabilities_to_sleep_stress_level(result_to_show['stress_probabilities']):.2f}")
    print(f"  Predicted sleep quality       : {result_to_show['predicted_sleep_quality']:.2f}")
    print(f"  Stress probabilities after    : {result_to_show['stress_probabilities']}")

    recommendation_df = pd.DataFrame(
        [
            {
                "current_stress_class": current_stress_class,
                "current_implied_sleep_stress_level": current_implied_stress_level,
                "current_sleep_duration_input": float(stress_profile.iloc[0]["Sleep_Duration"]),
                "current_sleep_quality_prediction": current_sleep_quality,
                "target_stress_class": target_stress_class,
                "target_sleep_quality": target_sleep_quality,
                "effective_sleep_quality_floor": effective_target_sleep_quality,
                "exact_goal_hit": best_result is not None,
                "recommendation_found": True,
                "recommended_activity_delta": candidate["activity_delta"],
                "recommended_screen_time_reduction": candidate["screen_delta"],
                "recommended_caffeine_reduction": candidate["caffeine_delta"],
                "recommended_alcohol_reduction": candidate["alcohol_delta"],
                "recommended_social_interaction_increase": candidate["social_delta"],
                "recommended_meditation_target": candidate["meditation_target"],
                "recommended_daily_steps_increase": candidate["steps_delta"],
                "predicted_stress_class_after_change": result_to_show["predicted_stress_class"],
                "predicted_implied_sleep_stress_level_after_change": stress_probabilities_to_sleep_stress_level(result_to_show["stress_probabilities"]),
                "predicted_sleep_quality_after_change": result_to_show["predicted_sleep_quality"],
                "high_stress_probability_reduction": result_to_show["high_probability_reduction"],
                "target_or_better_probability_gain": result_to_show["target_or_better_probability_gain"],
            }
        ]
    )
    recommendation_df.to_csv(RECOMMENDATION_OUTPUT_PATH, index=False)
    print(f"\nSaved combined recommendation to: {RECOMMENDATION_OUTPUT_PATH.resolve()}")


# ── App-facing API ────────────────────────────────────────────────────────────
# Lazy singleton — trained once per Python process, reused on every Streamlit rerun.
_combined_engine: dict | None = None


def _get_combined_engine() -> dict:
    global _combined_engine
    if _combined_engine is not None:
        return _combined_engine

    sleep_X, sleep_y = prepare_sleep_data()
    stress_X, stress_y = prepare_stress_data()
    model_bundle_path = Path(__file__).parent / "model.pkl"
    if model_bundle_path.exists():
        with open(model_bundle_path, "rb") as bundle_file:
            bundle = pickle.load(bundle_file)
        sleep_model = bundle["sleep_model"]
        stress_model = bundle["stress_model"]
    else:
        sleep_model = build_sleep_model(sleep_X)
        stress_model = build_stress_model(stress_X)

        sleep_X_train, _, sleep_y_train, _ = train_test_split(
            sleep_X, sleep_y, test_size=0.20, random_state=42,
        )
        stress_X_train, _, stress_y_train, _ = train_test_split(
            stress_X, stress_y, test_size=0.20, random_state=42, stratify=stress_y,
        )
        sleep_model.fit(sleep_X_train, sleep_y_train)
        stress_model.fit(stress_X_train, stress_y_train)

    _combined_engine = {
        "sleep_model":  sleep_model,
        "stress_model": stress_model,
        "sleep_X":      sleep_X,
        "stress_X":     stress_X,
    }
    return _combined_engine


def _ui_to_stress_df(stress_X: pd.DataFrame, sleep_duration, physical_activity,
                     screen_time, caffeine_intake, alcohol_intake, smoking_habit,
                     work_hours, social_interactions, meditation_practice,
                     exercise_type=None, **_) -> pd.DataFrame:
    """Map UI kwargs → stress model input DataFrame (Bed/Wake times are internal)."""
    medians = stress_X.median(numeric_only=True).to_dict()
    modes   = {c: stress_X[c].mode(dropna=True).iloc[0]
               for c in stress_X.select_dtypes(include="object").columns}
    row = {
        "Sleep_Duration":      float(sleep_duration),
        "Sleep_Quality":       medians.get("Sleep_Quality", 5.5),
        "Physical_Activity":   float(physical_activity),
        "Screen_Time":         float(screen_time),
        "Caffeine_Intake":     float(caffeine_intake),
        "Alcohol_Intake":      float(alcohol_intake),
        "Smoking_Habit":       1.0 if str(smoking_habit).strip().lower() in {"yes", "y", "1"} else 0.0,
        "Work_Hours":          float(work_hours),
        "Travel_Time":         medians.get("Travel_Time", 2.97),
        "Social_Interactions": float(social_interactions),
        "Meditation_Practice": 1.0 if str(meditation_practice).strip().lower() in {"yes", "y", "1"} else 0.0,
        "Exercise_Type":       str(exercise_type or modes.get("Exercise_Type", "Running")),
        "Marital_Status":      modes.get("Marital_Status", "Single"),
        "Wake_Up_Hour":        7.0,
        "Bed_Time_Hour":       31.0 - float(sleep_duration),
        "Time_In_Bed_Hours":   float(sleep_duration) + 1.0,
    }
    return pd.DataFrame([row])[stress_X.columns]


def _ui_to_sleep_df(sleep_X: pd.DataFrame, age, gender, sleep_duration,
                    physical_activity, daily_steps=None, **_) -> pd.DataFrame:
    """Map UI kwargs → sleep model input DataFrame (Stress Level set to NaN; filled later)."""
    modes   = {c: sleep_X[c].mode(dropna=True).iloc[0]
               for c in sleep_X.select_dtypes(include="object").columns}
    medians = sleep_X.median(numeric_only=True).to_dict()
    row = {
        "Gender":                  str(gender),
        "Age":                     float(age),
        "Sleep Duration":          float(sleep_duration),
        "Physical Activity Level": float(physical_activity) * 20.0,
        "Stress Level":            np.nan,
        "BMI Category":            modes.get("BMI Category", "Normal Weight"),
        "Daily Steps":             float(medians.get("Daily Steps", 7000.0) if daily_steps is None else daily_steps),
    }
    return pd.DataFrame([row])[sleep_X.columns]


def predict_combined_score(**kwargs) -> dict:
    """Return numeric sleep score (1–10 scale) and stress class for the given UI inputs."""
    engine = _get_combined_engine()
    s_df  = _ui_to_stress_df(stress_X=engine["stress_X"], **kwargs)
    sl_df = _ui_to_sleep_df(sleep_X=engine["sleep_X"], **kwargs)

    stress_class, stress_proba = predict_stress(engine["stress_model"], s_df)
    sl_df = update_sleep_profile_from_stress_and_actions(sl_df, stress_proba, 0.0, 0.0)
    sleep_score = predict_sleep_quality(engine["sleep_model"], sl_df)

    return {
        "sleep_score":  round(float(sleep_score), 2),
        "stress_class": stress_class,
        "stress_proba": stress_proba,
    }


def get_wellness_recommendations(**kwargs) -> dict:
    """
    Run the combined candidate search and return the best recommendation.

    Returns
    -------
    dict with keys:
        current_sleep_score   – float (1–10)
        current_stress_class  – str
        recommendation        – candidate dict or None
        predicted_sleep_after – float
        predicted_stress_after – str
    """
    engine       = _get_combined_engine()
    stress_model = engine["stress_model"]
    sleep_model  = engine["sleep_model"]
    stress_X     = engine["stress_X"]
    sleep_X      = engine["sleep_X"]

    s_df  = _ui_to_stress_df(stress_X=stress_X, **kwargs)
    sl_df = _ui_to_sleep_df(sleep_X=sleep_X, **kwargs)

    stress_class, stress_proba = predict_stress(stress_model, s_df)
    base_sl = update_sleep_profile_from_stress_and_actions(sl_df.copy(), stress_proba, 0.0, 0.0)
    current_sleep = predict_sleep_quality(sleep_model, base_sl)

    target_stress  = default_target_stress_class(stress_class)
    target_rank    = STRESS_TARGET_ORDER.index(target_stress)
    current_rank   = STRESS_TARGET_ORDER.index(stress_class)
    stress_sat     = current_rank <= target_rank
    cur_high_p     = stress_proba.get("High", 0.0)
    cur_target_p   = sum(stress_proba.get(l, 0.0) for l in STRESS_TARGET_ORDER[:target_rank + 1])
    eff_target     = current_sleep + 0.5

    best = None
    best_imp = None
    closest = None

    for candidate in generate_combined_candidates(s_df):
        cs_df = apply_stress_candidate(s_df, candidate)
        pred_stress, pred_proba = predict_stress(stress_model, cs_df)
        pred_rank = STRESS_TARGET_ORDER.index(pred_stress)

        csl_df = update_sleep_profile_from_stress_and_actions(
            sl_df.copy(), pred_proba,
            float(candidate["activity_delta"]) * 10.0,
            float(candidate["steps_delta"]),
        )
        pred_sleep = predict_sleep_quality(sleep_model, csl_df)
        change_score, n_changed = candidate_score(candidate, s_df)

        if not allow_candidate_under_policy(
            candidate, n_changed, pred_sleep, current_sleep,
            pred_rank, target_rank, stress_sat,
        ):
            continue
        if n_changed == 0:
            continue

        high_red    = cur_high_p - pred_proba.get("High", 0.0)
        target_gain = (sum(pred_proba.get(l, 0.0) for l in STRESS_TARGET_ORDER[:target_rank + 1])
                       - cur_target_p)

        result = {
            "candidate":    candidate,
            "pred_stress":  pred_stress,
            "pred_sleep":   round(float(pred_sleep), 2),
            "high_red":     high_red,
            "target_gain":  target_gain,
            "change_score": change_score,
            "n_changed":    n_changed,
            "power_score":  candidate_power_score(
                candidate=candidate,
                original_stress_profile=s_df,
                current_sleep_quality=current_sleep,
                predicted_sleep_quality=pred_sleep,
                current_high_probability=cur_high_p,
                predicted_probabilities=pred_proba,
                current_target_or_better_probability=cur_target_p,
                target_rank=target_rank,
            ),
        }

        closeness_key = (
            max(0, pred_rank - target_rank),
            round(max(0.0, eff_target - pred_sleep), 4),
            -round(result["power_score"], 6),
            -round(target_gain, 6), -round(high_red, 6),
            change_score, n_changed,
        )
        if closest is None or closeness_key < closest["closeness_key"]:
            result["closeness_key"] = closeness_key
            closest = result

        if stress_sat:
            imp_key  = (-round(pred_sleep - current_sleep, 6), n_changed, change_score)
            qualifies = (n_changed > 0 and pred_rank <= target_rank
                         and pred_sleep - current_sleep >= MIN_SLEEP_IMPROVEMENT_WHEN_STRESS_ALREADY_SATISFIED)
        else:
            imp_key  = (-round(target_gain, 6), -round(high_red, 6),
                        -round(pred_sleep, 6), change_score, n_changed)
            qualifies = n_changed > 0 and (target_gain > 0 or high_red > 0 or pred_sleep > current_sleep)

        if qualifies:
            if best_imp is None or (
                -round(result["power_score"], 6),
                *imp_key,
            ) < (
                -round(best_imp["power_score"], 6),
                *best_imp["imp_key"],
            ):
                result["imp_key"] = imp_key
                best_imp = result

        if pred_rank > target_rank or pred_sleep < eff_target:
            continue

        if best is None:
            best = result
            continue

        cur_key  = (pred_rank, -round(result["power_score"], 6), n_changed, change_score, -pred_sleep)
        best_key = (STRESS_TARGET_ORDER.index(best["pred_stress"]),
                    -round(best["power_score"], 6),
                    best["n_changed"], best["change_score"], -best["pred_sleep"])
        if cur_key < best_key:
            best = result

    chosen = best or best_imp or closest
    if chosen is None:
        return {
            "current_sleep_score":    round(float(current_sleep), 2),
            "current_stress_class":   stress_class,
            "recommendation":         None,
            "predicted_sleep_after":  round(float(current_sleep), 2),
            "predicted_stress_after": stress_class,
        }

    return {
        "current_sleep_score":    round(float(current_sleep), 2),
        "current_stress_class":   stress_class,
        "recommendation":         chosen["candidate"],
        "predicted_sleep_after":  chosen["pred_sleep"],
        "predicted_stress_after": chosen["pred_stress"],
    }


if __name__ == "__main__":
    main()
