"""
predictor.py  —  SleepIQ two-stage core engine

Stage 1: Stress classifier  (lifestyle → Low / Medium / High)
Stage 2: Sleep classifier   (lifestyle + stress level → Poor / Moderate / Good)

Models are sklearn Pipelines trained by create_model_pkl.py from
combined_wellness_recommender.py.  Run create_model_pkl.py once to produce
model.pkl, then this module loads it for fast inference.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from combined_wellness_recommender import stress_probabilities_to_sleep_stress_level
from sleep_wrapper import SleepClassifierWrapper  # noqa: F401 — needed for pickle to resolve the class

# ── Load models ───────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent / "model.pkl"
with open(_MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

STRESS_MODEL          = _bundle["stress_model"]
STRESS_FEATURE_NAMES  = _bundle["stress_feature_names"]
STRESS_LABEL_MAP      = _bundle["stress_label_map"]
STRESS_INV_LABEL      = _bundle["stress_inv_label"]
STRESS_FEAT_IMP       = _bundle["stress_feature_importances"]
STRESS_METRICS        = _bundle.get("stress_metrics", {})

SLEEP_MODEL           = _bundle["sleep_model"]
SLEEP_FEATURE_NAMES   = _bundle["sleep_feature_names"]
SLEEP_LABEL_MAP       = _bundle["sleep_label_map"]
SLEEP_INV_LABEL       = _bundle["sleep_inv_label"]
SLEEP_FEAT_IMP        = _bundle["sleep_feature_importances"]
SLEEP_METRICS         = _bundle.get("sleep_metrics", {})

_STRESS_MEDIANS = _bundle.get("stress_X_medians", {})
_STRESS_MODES   = _bundle.get("stress_X_modes", {})
_SLEEP_MEDIANS  = _bundle.get("sleep_X_medians", {})
_SLEEP_MODES    = _bundle.get("sleep_X_modes", {})

# ── UI constants ──────────────────────────────────────────────────────────────
GENDERS            = ["Male", "Female"]
SMOKING_OPTIONS    = ["No", "Yes"]
MEDITATION_OPTIONS = ["No", "Yes"]
SLEEP_COLORS = {"Poor": "#e74c3c", "Moderate": "#f39c12", "Good": "#2ecc71"}
SLEEP_EMOJI  = {"Poor": "😴",      "Moderate": "🌙",      "Good": "⭐"}

STRESS_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
STRESS_EMOJI  = {"Low": "😌",      "Medium": "😐",       "High": "😤"}

# Defaults for stress-model fields not exposed in the UI
_STRESS_HIDDEN = {
    "Sleep_Quality":  6.5,
    "Travel_Time":    1.0,
    "Wake_Up_Hour":   7.0,
    "Exercise_Type":  _STRESS_MODES.get("Exercise_Type",    "Running"),
    "Marital_Status": _STRESS_MODES.get("Marital_Status",   "Single"),
}

# Defaults for sleep-model fields not exposed in the UI
_SLEEP_HIDDEN = {
    "BMI Category": _SLEEP_MODES.get("BMI Category",  "Normal Weight"),
    "Daily Steps":  _SLEEP_MEDIANS.get("Daily Steps",  7000.0),
}

# Optimizer levers (only lifestyle variables the user can actually change)
# (key-in-stress-feature-space, direction, step, hard_min, hard_max, quick_win_cap)
LEVERS = [
    ("Sleep_Duration",       +1,  0.25,  3.0,  10.0,  1.5),
    ("Physical_Activity",    +1,  5,     0,    120,   30),
    ("Screen_Time",          -1,  0.5,   0.0,  12.0,  3.0),
    ("Caffeine_Intake",      -1,  1,     0,    10,    3),
    ("Social_Interactions",  +1,  0.5,   0.0,  10.0,  3.0),
    ("Daily_Steps",          +1,  500,   3000, 10000, 2500),
]
TIPS = {
    "Sleep_Duration":      "Sleep Sleep Sleep! Aim for 7-9 hours per night for optimal health.",
    "Physical_Activity":   "A 20-min walk counts! Morning movement has the biggest sleep payoff.",
    "Screen_Time":         "Use an app blocker to limit evening screen time, which can disrupt your circadian rhythm.",
    "Caffeine_Intake":     "Avoid caffeine after 2 pm; it has a 5-hr half-life in your body.",
    "Social_Interactions": "Join a club, class, or online community to boost your social time.",
    "Daily_Steps":         "Add steady walking through the day to build a sustainable step-count increase.",
}
LEVER_LABELS = {
    "Sleep_Duration":      "Sleep Duration (hrs)",
    "Physical_Activity":   "Physical Activity (min/day)",
    "Screen_Time":         "Screen Time (hrs/day)",
    "Caffeine_Intake":     "Caffeine Intake (cups/day)",
    "Social_Interactions": "Social Interactions (hrs/day)",
    "Daily_Steps":         "Daily Steps",
}

LEVER_HEALTH_TARGETS = {
    "Sleep_Duration":      {"direction": "up", "target": 7.5},
    "Physical_Activity":   {"direction": "up", "target": 45.0},
    "Screen_Time":         {"direction": "down", "target": 3.0},
    "Caffeine_Intake":     {"direction": "down", "target": 2.0},
    "Social_Interactions": {"direction": "up", "target": 3.0},
    "Daily_Steps":         {"direction": "up", "target": 8000.0},
}

QUICK_MINOR_CAPS = {
    "Sleep_Duration": 0.75,
    "Physical_Activity": 15.0,
    "Screen_Time": 1.0,
    "Caffeine_Intake": 1.0,
    "Social_Interactions": 1.0,
    "Daily_Steps": 1500.0,
}

LONG_TERM_TARGETS = {
    "Sleep_Duration": 8.5,
    "Physical_Activity": 60.0,
    "Screen_Time": 2.0,
    "Caffeine_Intake": 1.0,
    "Social_Interactions": 4.0,
    "Daily_Steps": 9000.0,
}

STRESS_EXPLAIN_INPUT_FEATURES = {
    "Sleep_Duration",
    "Physical_Activity",
    "Screen_Time",
    "Caffeine_Intake",
    "Alcohol_Intake",
    "Smoking_Habit",
    "Work_Hours",
    "Social_Interactions",
    "Meditation_Practice",
}

SLEEP_EXPLAIN_INPUT_FEATURES = {
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Daily Steps",
    "Stress Level",
}
def _estimated_sleep_quality(sleep_duration: float) -> float:
    duration = float(sleep_duration)
    if duration <= 5.0:
        return 4.5
    if duration <= 6.0:
        return 5.5
    if duration <= 7.0:
        return 6.5
    if duration <= 8.0:
        return 7.5
    return 8.2
def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stress_wellness_index(fv: pd.DataFrame) -> float:
    row = fv.iloc[0]

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

    weighted = (
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
    return _clip01(weighted)


def _stress_wellness_probabilities(fv: pd.DataFrame) -> np.ndarray:
    wellness = _stress_wellness_index(fv)
    high = _clip01((0.48 - wellness) / 0.30)
    low = _clip01((wellness - 0.58) / 0.25)
    medium = max(0.0, 1.0 - high - low)
    probs = np.array([high, low, medium], dtype=float)
    total = probs.sum()
    return probs / total if total > 0 else np.array([0.33, 0.33, 0.34], dtype=float)


def _blend_stress_probabilities(raw_proba: np.ndarray, fv: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(raw_proba, dtype=float)
    wellness = _stress_wellness_probabilities(fv)
    blended = (0.55 * raw) + (0.45 * wellness)
    total = blended.sum()
    return blended / total if total > 0 else raw


def _calibrated_stress_label(proba: np.ndarray, fv: pd.DataFrame) -> str:
    blended = _blend_stress_probabilities(proba, fv)
    scores = {STRESS_INV_LABEL[i]: float(blended[i]) for i in range(len(blended))}
    high = scores.get("High", 0.0)
    medium = scores.get("Medium", 0.0)
    low = scores.get("Low", 0.0)
    wellness = _stress_wellness_index(fv)

    if wellness >= 0.68 and high < 0.62:
        return "Low" if low >= 0.30 else "Medium"
    if wellness >= 0.58 and high < 0.68:
        return "Medium"
    if high >= 0.60 and (high - medium) >= 0.12 and wellness < 0.52:
        return "High"
    if low >= 0.46 and (low - medium) >= 0.08:
        return "Low"
    if medium >= 0.32:
        return "Medium"
    return max(scores, key=scores.get)

def build_stress_fv(smoking_habit, meditation_practice,
                    sleep_duration, physical_activity, screen_time,
                    caffeine_intake, alcohol_intake, work_hours,
                    social_interactions, exercise_type=None, **_) -> pd.DataFrame:
    """Build a one-row DataFrame matching the stress Pipeline's expected raw input."""
    row = {
        "Sleep_Duration":      float(sleep_duration),
        "Sleep_Quality":       _estimated_sleep_quality(sleep_duration),
        "Physical_Activity":   float(physical_activity),
        "Screen_Time":         float(screen_time),
        "Caffeine_Intake":     float(caffeine_intake),
        "Alcohol_Intake":      float(alcohol_intake),
        "Smoking_Habit":       1.0 if smoking_habit == "Yes" else 0.0,
        "Work_Hours":          float(work_hours),
        "Travel_Time":         float(_STRESS_HIDDEN["Travel_Time"]),
        "Social_Interactions": float(social_interactions),
        "Meditation_Practice": 1.0 if meditation_practice == "Yes" else 0.0,
        "Exercise_Type":       str(exercise_type or _STRESS_HIDDEN["Exercise_Type"]),
        "Marital_Status":      str(_STRESS_HIDDEN["Marital_Status"]),
        "Wake_Up_Hour":        float(_STRESS_HIDDEN["Wake_Up_Hour"]),
        "Bed_Time_Hour":       float(31.0 - sleep_duration),
        "Time_In_Bed_Hours":   float(sleep_duration + 1.0),
    }
    return pd.DataFrame([row])[STRESS_FEATURE_NAMES]


def build_sleep_fv(stress_proba: np.ndarray,
                   age, gender, sleep_duration, physical_activity, daily_steps=None, **_) -> pd.DataFrame:
    """Build a one-row DataFrame matching the sleep Pipeline's expected raw input.

    Converts the stress probability vector into a numeric Stress Level via the
    calibrated bridge function from combined_wellness_recommender.
    """
    stress_proba_dict = {
        STRESS_INV_LABEL[i]: float(p) for i, p in enumerate(stress_proba)
    }
    stress_level = stress_probabilities_to_sleep_stress_level(stress_proba_dict)

    row = {
        "Gender":                  str(gender),
        "Age":                     float(age),
        "Sleep Duration":          float(sleep_duration),
        "Physical Activity Level": float(physical_activity) * 20.0,
        "Stress Level":            float(stress_level),
        "BMI Category":            str(_SLEEP_HIDDEN["BMI Category"]),
        "Daily Steps":             float(_SLEEP_HIDDEN["Daily Steps"] if daily_steps is None else daily_steps),
    }
    return pd.DataFrame([row])[SLEEP_FEATURE_NAMES]


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_stress(fv: pd.DataFrame) -> dict:
    raw_proba = STRESS_MODEL.predict_proba(fv)[0]
    proba = _blend_stress_probabilities(raw_proba, fv)
    label = _calibrated_stress_label(raw_proba, fv)
    idx = max((i for i in range(len(proba)) if STRESS_INV_LABEL[i] == label), key=lambda i: proba[i])
    return {
        "label":         label,
        "probabilities": {STRESS_INV_LABEL[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "confidence":    round(float(proba[idx]), 4),
        "color":         STRESS_COLORS[label],
        "emoji":         STRESS_EMOJI[label],
    }


def predict_sleep(fv: pd.DataFrame) -> dict:
    proba = SLEEP_MODEL.predict_proba(fv)[0]
    idx   = int(np.argmax(proba))
    label = SLEEP_INV_LABEL[idx]
    return {
        "label":         label,
        "probabilities": {SLEEP_INV_LABEL[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "confidence":    round(float(proba[idx]), 4),
        "color":         SLEEP_COLORS[label],
        "emoji":         SLEEP_EMOJI[label],
    }


def predict_both(**kwargs) -> dict:
    """Full two-stage prediction from raw lifestyle inputs."""
    stress_fv     = build_stress_fv(**kwargs)
    stress_result = predict_stress(stress_fv)
    stress_proba  = np.array([stress_result["probabilities"][STRESS_INV_LABEL[i]] for i in range(len(STRESS_INV_LABEL))], dtype=float)
    sleep_fv      = build_sleep_fv(stress_proba, **kwargs)
    sleep_result  = predict_sleep(sleep_fv)
    return {"stress": stress_result, "sleep": sleep_result}


# ── Explain ───────────────────────────────────────────────────────────────────
def _clean_name(fname: str) -> str:
    return (fname
            .replace("Smoking_Habit", "Smoking")
            .replace("Meditation_Practice", "Meditation")
            .replace("Physical_Activity", "Physical Activity")
            .replace("Social_Interactions", "Social Interactions")
            .replace("Sleep_Duration", "Sleep Duration")
            .replace("Screen_Time", "Screen Time")
            .replace("Caffeine_Intake", "Caffeine Intake")
            .replace("Alcohol_Intake", "Alcohol Intake")
            .replace("Work_Hours", "Work Hours")
            .replace("Travel_Time", "Travel Time")
            .replace("Sleep_Quality", "Sleep Quality")
            .replace("Wake_Up_Hour", "Wake Up Hour")
            .replace("Bed_Time_Hour", "Bed Time Hour")
            .replace("Time_In_Bed_Hours", "Time In Bed")
            .replace("Physical Activity Level", "Physical Activity Level")
            .replace("Daily Steps", "Daily Steps")
            .replace("BMI Category", "BMI Category")
            .replace("Stress Level", "Stress Level")
            .replace("Sleep Duration", "Sleep Duration"))


def _perturbation_explain(model, fv: pd.DataFrame, inv_label: dict,
                          allowed_features: set[str] | None = None,
                          top_n: int = 5) -> dict:
    """Estimate feature contributions by zeroing one numeric feature at a time."""
    base_proba = model.predict_proba(fv)[0]
    pred_class = int(np.argmax(base_proba))
    label      = inv_label[pred_class]

    contribs = []
    for fname in fv.columns:
        if allowed_features is not None and fname not in allowed_features:
            continue
        # Only perturb numeric columns; categoricals can't meaningfully be zeroed
        if not pd.api.types.is_numeric_dtype(fv[fname]):
            continue
        perturbed = fv.copy()
        perturbed[fname] = 0.0
        delta = float(base_proba[pred_class] - model.predict_proba(perturbed)[0][pred_class])
        contribs.append({
            "feature":   fname,
            "label":     _clean_name(fname),
            "impact":    delta,
            "raw_value": float(fv.iloc[0][fname]),
        })

    df  = pd.DataFrame(contribs)
    pos = df[df.impact > 0.005].nlargest(top_n, "impact").to_dict("records")
    neg = df[df.impact < -0.005].nsmallest(top_n, "impact").to_dict("records")

    if pos:
        top_pos = pos[0]["label"]
        top_neg = neg[0]["label"] if neg else None
        summary = (
            f"Your <b>{top_pos}</b> is the strongest driver of this prediction."
            + (f" However, <b>{top_neg}</b> is partially working against it." if top_neg else "")
        )
    else:
        summary = "Multiple factors are contributing roughly equally to this prediction."

    return {"label": label, "supporting": pos, "opposing": neg, "summary": summary}


def explain_stress(**kwargs) -> dict:
    fv = build_stress_fv(**kwargs)
    explanation = _perturbation_explain(
        STRESS_MODEL, fv, STRESS_INV_LABEL, allowed_features=STRESS_EXPLAIN_INPUT_FEATURES
    )
    explanation["label"] = predict_stress(fv)["label"]
    return explanation


def explain_sleep(**kwargs) -> dict:
    stress_fv    = build_stress_fv(**kwargs)
    stress_result = predict_stress(stress_fv)
    stress_proba = np.array([stress_result["probabilities"][STRESS_INV_LABEL[i]] for i in range(len(STRESS_INV_LABEL))], dtype=float)
    sleep_fv     = build_sleep_fv(stress_proba, **kwargs)
    explanation = _perturbation_explain(
        SLEEP_MODEL, sleep_fv, SLEEP_INV_LABEL, allowed_features=SLEEP_EXPLAIN_INPUT_FEATURES, top_n=6
    )

    step_row = next((row for row in explanation["supporting"] + explanation["opposing"]
                     if row["feature"] == "Daily Steps"), None)
    if step_row is None and "Daily Steps" in sleep_fv.columns:
        base_proba = SLEEP_MODEL.predict_proba(sleep_fv)[0]
        pred_idx = int(np.argmax(base_proba))
        perturbed = sleep_fv.copy()
        perturbed["Daily Steps"] = 0.0
        step_delta = float(base_proba[pred_idx] - SLEEP_MODEL.predict_proba(perturbed)[0][pred_idx])
        if abs(step_delta) >= 0.0025:
            injected = {
                "feature": "Daily Steps",
                "label": "Daily Steps",
                "impact": step_delta,
                "raw_value": float(sleep_fv.iloc[0]["Daily Steps"]),
            }
            target_list = explanation["supporting"] if step_delta > 0 else explanation["opposing"]
            target_list.append(injected)
            target_list.sort(key=lambda row: abs(float(row["impact"])), reverse=True)
            del target_list[6:]

    if not explanation["summary"].endswith('predicted stress level from Stage 1.'):
        explanation["summary"] += ' Sleep also uses your predicted stress level from Stage 1.'
    return explanation


# ── Optimizer ─────────────────────────────────────────────────────────────────
_LEVER_TO_KW = {
    "Sleep_Duration":      "sleep_duration",
    "Physical_Activity":   "physical_activity",
    "Screen_Time":         "screen_time",
    "Caffeine_Intake":     "caffeine_intake",
    "Social_Interactions": "social_interactions",
    "Daily_Steps":         "daily_steps",
}


def _params_to_results(fixed: dict, levers: dict) -> tuple[dict, dict]:
    lever_kw = {_LEVER_TO_KW[k]: v for k, v in levers.items()}
    merged   = {**fixed, **lever_kw}
    stress_fv     = build_stress_fv(**merged)
    stress_result = predict_stress(stress_fv)
    stress_proba  = np.array([stress_result["probabilities"][STRESS_INV_LABEL[i]] for i in range(len(STRESS_INV_LABEL))], dtype=float)
    sleep_fv      = build_sleep_fv(stress_proba, **merged)
    sleep_result  = predict_sleep(sleep_fv)
    return stress_result, sleep_result


def _build_percent_caps(lever_vals: dict, limit_fraction: float = 0.75) -> dict:
    caps = {}
    for key, _, step, lo, hi, _ in LEVERS:
        original = float(lever_vals[key])
        baseline = abs(original) * limit_fraction
        if baseline <= 0:
            baseline = step
        max_possible = max(abs(hi - original), abs(original - lo))
        caps[key] = min(max_possible, max(step, baseline))
    return caps


def _lever_effort(key: str, original_value: float, new_value: float) -> float:
    delta = abs(float(new_value) - float(original_value))
    if key == "Daily_Steps":
        return delta / max(1000.0, abs(float(original_value)) * 0.25)
    return delta / max(1.0, abs(float(original_value)) * 0.25)


def _lever_need_score(key: str, current_value: float) -> float:
    cfg = LEVER_HEALTH_TARGETS[key]
    target = float(cfg["target"])
    current = float(current_value)
    if cfg["direction"] == "up":
        deficit = max(0.0, target - current)
        return deficit / max(1.0, target)
    excess = max(0.0, current - target)
    return excess / max(1.0, target)


def _stress_badness(stress_result: dict) -> float:
    probabilities = stress_result.get("probabilities", {})
    high = float(probabilities.get("High", 0.0))
    medium = float(probabilities.get("Medium", 0.0))
    label_penalty = {"Low": 0.0, "Medium": 0.2, "High": 0.45}.get(stress_result.get("label"), 0.0)
    return high + (0.35 * medium) + label_penalty


def _state_score(stress_result: dict, sleep_result: dict, sleep_target: str, mode: str) -> float:
    target_prob = float(sleep_result["probabilities"].get(sleep_target, 0.0))
    reached_bonus = 1.25 if sleep_result["label"] == sleep_target else 0.0
    stress_improvement = 1.0 - _stress_badness(stress_result)
    if mode == "quick":
        return (2.2 * target_prob) + reached_bonus + (0.6 * stress_improvement)
    stress_label_bonus = {"Low": 0.8, "Medium": 0.3, "High": 0.0}.get(stress_result["label"], 0.0)
    return (2.8 * target_prob) + (1.4 * reached_bonus) + (1.8 * stress_improvement) + stress_label_bonus


def _run_optimizer(lever_vals: dict, fixed: dict,
                   sleep_target: str, max_deltas, max_recommendations: int = 3,
                   mode: str = "quick") -> dict:
    original = lever_vals.copy()
    current  = lever_vals.copy()

    for _ in range(80):
        current_stress, sleep_r = _params_to_results(fixed, current)
        if sleep_r["label"] == sleep_target and (mode == "quick" or current_stress["label"] != "High"):
            break
        current_target_prob = sleep_r["probabilities"].get(sleep_target, 0.0)
        current_stress_badness = _stress_badness(current_stress)
        current_needs = {key: _lever_need_score(key, current[key]) for key in current}
        max_need = max(current_needs.values()) if current_needs else 0.0
        best_score, best_gain, best_key, best_val = -1, -1, None, None
        for key, direction, step, lo, hi, _ in LEVERS:
            new_val = round(current[key] + direction * step, 2)
            if not (lo <= new_val <= hi):
                continue
            if max_deltas and abs(new_val - original[key]) > max_deltas.get(key, 9e9):
                continue
            if max_need > 0.05 and current_needs.get(key, 0.0) <= 0.0:
                continue
            trial_stress, trial_sleep = _params_to_results(fixed, {**current, key: new_val})
            target_prob = trial_sleep["probabilities"].get(sleep_target, 0.0)
            gain = max(0.0, target_prob - current_target_prob)
            reach_bonus = 1.0 if trial_sleep["label"] == sleep_target else 0.0
            effort = _lever_effort(key, current[key], new_val)
            need = current_needs.get(key, 0.0)
            need_bonus = 1.0 + (3.0 * need)
            if max_need > 0.05 and need < max_need * 0.35:
                need_bonus *= 0.2
            stress_gain = max(0.0, current_stress_badness - _stress_badness(trial_stress))
            if mode == "quick":
                score = ((1.8 * gain) + (0.7 * stress_gain) + reach_bonus) * need_bonus / max(0.25, effort)
            else:
                stress_label_bonus = 0.0
                if current_stress["label"] == "High" and trial_stress["label"] != "High":
                    stress_label_bonus = 1.25
                elif current_stress["label"] == "Medium" and trial_stress["label"] == "Low":
                    stress_label_bonus = 0.8
                score = (
                    (2.6 * gain)
                    + (1.8 * stress_gain)
                    + (1.4 * reach_bonus)
                    + stress_label_bonus
                ) * need_bonus / max(0.35, effort ** 0.85)
            if score > best_score or (score == best_score and gain > best_gain):
                best_score, best_gain, best_key, best_val = score, gain, key, new_val
        if best_key is None:
            break
        current[best_key] = best_val

    final_stress, final_sleep = _params_to_results(fixed, current)
    recs = []
    for key, *_ in LEVERS:
        orig_val, new_val = original[key], current[key]
        if orig_val == new_val:
            continue
        recs.append({
            "feature": LEVER_LABELS[key], "key": key,
            "from": orig_val, "to": new_val,
            "delta": round(new_val - orig_val, 2),
            "tip":   TIPS[key],
            "effort": _lever_effort(key, orig_val, new_val),
        })
    recs = sorted(recs, key=lambda rec: (rec["effort"], -abs(float(rec["delta"]))))[:max_recommendations]
    for rec in recs:
        rec.pop("effort", None)
    return {
        "recommendations": recs,
        "new_sleep":       final_sleep,
        "new_stress":      final_stress,
        "reached_target":  final_sleep["label"] == sleep_target,
        "score":           float(best_score),
    }


def _enforce_stronger_than_quick(quick: dict, full: dict) -> dict:
    quick_recs = {rec["feature"]: rec for rec in quick.get("recommendations", [])}
    updated = []
    changed = False
    for rec in full.get("recommendations", []):
        qrec = quick_recs.get(rec.get("feature"))
        if qrec and abs(float(rec.get("delta", 0.0))) < abs(float(qrec.get("delta", 0.0))):
            updated.append({**rec, "to": qrec.get("to"), "delta": qrec.get("delta")})
            changed = True
        else:
            updated.append(rec)
    if changed:
        full = {**full, "recommendations": updated}
    return full


def _run_strong_optimizer(lever_vals: dict, fixed: dict,
                          sleep_target: str, max_deltas, max_recommendations: int = 3,
                          avoid_features: set[str] | None = None) -> dict:
    original = lever_vals.copy()
    base_stress, base_sleep = _params_to_results(fixed, original)
    beam = [(
        _state_score(base_stress, base_sleep, sleep_target, "strong"),
        original.copy(),
        base_stress,
        base_sleep,
    )]
    seen = {tuple(sorted(original.items()))}

    for _ in range(10):
        candidates = []
        for _, state, _, _ in beam:
            for key, direction, step, lo, hi, _ in LEVERS:
                if avoid_features and key in avoid_features:
                    continue
                new_val = round(state[key] + direction * step, 2)
                if not (lo <= new_val <= hi):
                    continue
                if max_deltas and abs(new_val - original[key]) > max_deltas.get(key, 9e9):
                    continue
                new_state = {**state, key: new_val}
                state_key = tuple(sorted(new_state.items()))
                if state_key in seen:
                    continue
                seen.add(state_key)
                trial_stress, trial_sleep = _params_to_results(fixed, new_state)
                score = _state_score(trial_stress, trial_sleep, sleep_target, "strong")
                candidates.append((score, new_state, trial_stress, trial_sleep))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = candidates[:8]
        if any(sleep["label"] == sleep_target and stress["label"] != "High" for _, _, stress, sleep in beam):
            break

    avoid_features = avoid_features or set()
    def _beam_choice(item):
        score, state, stress, sleep = item
        changed = {k for k in state if state[k] != original[k]}
        overlap = len(changed & avoid_features)
        distinct = len(changed - avoid_features)
        stress_bonus = 0.2 if stress["label"] == "Low" else 0.08 if stress["label"] == "Medium" else 0.0
        return (score + (0.18 * distinct) - (0.12 * overlap) + stress_bonus, score)

    best_score, best_state, final_stress, final_sleep = max(beam, key=_beam_choice)
    recs = []
    for key, *_ in LEVERS:
        orig_val, new_val = original[key], best_state[key]
        if orig_val == new_val:
            continue
        recs.append({
            "feature": LEVER_LABELS[key], "key": key,
            "from": orig_val, "to": new_val,
            "delta": round(new_val - orig_val, 2),
            "tip":   TIPS[key],
            "effort": _lever_effort(key, orig_val, new_val),
        })
    recs = sorted(recs, key=lambda rec: (-abs(float(rec["delta"])), rec["effort"]))[:max_recommendations]
    for rec in recs:
        rec.pop("effort", None)
    return {
        "recommendations": recs,
        "new_sleep":       final_sleep,
        "new_stress":      final_stress,
        "reached_target":  final_sleep["label"] == sleep_target,
        "score":           float(best_score),
    }


def _lever_step_and_bounds(key: str) -> tuple[float, float, float]:
    for lever_key, _, step, lo, hi, _ in LEVERS:
        if lever_key == key:
            return float(step), float(lo), float(hi)
    raise KeyError(key)


def _move_toward_target(key: str, current_value: float, target_value: float, max_delta: float) -> float:
    step, lo, hi = _lever_step_and_bounds(key)
    current = float(current_value)
    target = float(target_value)
    allowed = max(float(step), float(max_delta))
    if target > current:
        moved = min(target, current + allowed)
    else:
        moved = max(target, current - allowed)
    moved = min(hi, max(lo, moved))
    if step >= 1.0:
        moved = round(moved)
    else:
        moved = round(round(moved / step) * step, 2)
    return float(min(hi, max(lo, moved)))


def _feature_priority(key: str, fixed: dict, levers: dict, sleep_target: str) -> float:
    current = float(levers[key])
    need = _lever_need_score(key, current)
    if need <= 0:
        return 0.0
    step, lo, hi = _lever_step_and_bounds(key)
    direction = 1.0 if LEVER_HEALTH_TARGETS[key]["direction"] == "up" else -1.0
    trial = round(current + (direction * step), 2)
    if not (lo <= trial <= hi):
        return 0.0
    base_stress, base_sleep = _params_to_results(fixed, levers)
    trial_state = {**levers, key: trial}
    trial_stress, trial_sleep = _params_to_results(fixed, trial_state)
    base_score = _state_score(base_stress, base_sleep, sleep_target, "strong")
    trial_score = _state_score(trial_stress, trial_sleep, sleep_target, "strong")
    return max(0.0, trial_score - base_score) + (1.35 * need)


def _build_plan_from_targets(levers: dict, fixed: dict, selected_keys: list[str], target_map: dict[str, float], max_delta_map: dict[str, float], sleep_target: str) -> dict:
    original = levers.copy()
    planned = levers.copy()
    for key in selected_keys:
        planned[key] = _move_toward_target(key, planned[key], target_map[key], max_delta_map[key])
    final_stress, final_sleep = _params_to_results(fixed, planned)
    recs = []
    for key in selected_keys:
        orig_val, new_val = original[key], planned[key]
        if orig_val == new_val:
            continue
        recs.append({
            "feature": LEVER_LABELS[key], "key": key,
            "from": orig_val, "to": new_val,
            "delta": round(new_val - orig_val, 2),
            "tip": TIPS[key],
        })
    return {
        "recommendations": recs,
        "new_sleep": final_sleep,
        "new_stress": final_stress,
        "reached_target": final_sleep["label"] == sleep_target,
        "score": float(_state_score(final_stress, final_sleep, sleep_target, "strong")),
    }


def optimize(sleep_target="Good", **kwargs) -> dict:
    fixed = {k: kwargs[k] for k in
             ("age", "gender", "smoking_habit", "meditation_practice",
              "alcohol_intake", "work_hours")}
    levers = {
        "Sleep_Duration": kwargs["sleep_duration"],
        "Physical_Activity": kwargs["physical_activity"],
        "Screen_Time": kwargs["screen_time"],
        "Caffeine_Intake": kwargs["caffeine_intake"],
        "Social_Interactions": kwargs["social_interactions"],
        "Daily_Steps": kwargs["daily_steps"],
    }

    base_stress, base_sleep = _params_to_results(fixed, levers)

    if base_sleep["label"] == sleep_target:
        base_quick_score = float(_state_score(base_stress, base_sleep, sleep_target, "quick"))
        base_full_score = float(_state_score(base_stress, base_sleep, sleep_target, "strong"))
        return {
            "already_optimal": True,
            "base_sleep": base_sleep,
            "base_stress": base_stress,
            "quick_wins": {"recommendations": [], "new_sleep": base_sleep,
                             "new_stress": base_stress, "reached_target": True, "score": base_quick_score},
            "full_fix": {"recommendations": [], "new_sleep": base_sleep,
                           "new_stress": base_stress, "reached_target": True, "score": base_full_score},
        }

    percent_caps = _build_percent_caps(levers, limit_fraction=1.5)
    priority_order = sorted(
        levers.keys(),
        key=lambda key: (_feature_priority(key, fixed, levers, sleep_target), _lever_need_score(key, levers[key])),
        reverse=True,
    )
    selected_keys = [key for key in priority_order if _lever_need_score(key, levers[key]) > 0][:3]
    if len(selected_keys) < 3:
        selected_keys = priority_order[:3]

    quick_caps = {key: 0.0 for key in levers}
    for key in selected_keys:
        quick_caps[key] = min(QUICK_MINOR_CAPS.get(key, percent_caps[key]), percent_caps[key])

    quick = _run_optimizer(
        levers, fixed, sleep_target, max_deltas=quick_caps, max_recommendations=3, mode="quick"
    )

    full_caps = {key: 0.0 for key in levers}
    for key in selected_keys:
        realistic_delta = abs(float(LONG_TERM_TARGETS.get(key, levers[key])) - float(levers[key]))
        step = _lever_step_and_bounds(key)[0]
        full_caps[key] = max(
            quick_caps[key] + step,
            min(percent_caps[key], max(realistic_delta, quick_caps[key] * 1.75))
        )

    full = _run_strong_optimizer(
        levers, fixed, sleep_target, max_deltas=full_caps, max_recommendations=3
    )

    if full.get("score", -1) < quick.get("score", -1):
        full = _run_optimizer(
            levers, fixed, sleep_target, max_deltas=full_caps, max_recommendations=3, mode="strong"
        )
    if full.get("score", -1) < quick.get("score", -1):
        full = {**quick, "score": quick.get("score", -1)}

    return {
        "already_optimal": False,
        "base_sleep": base_sleep,
        "base_stress": base_stress,
        "quick_wins": quick,
        "full_fix": full,
    }


