import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from predictor import (
    predict_both,
    optimize,
    explain_stress,
    explain_sleep,
    GENDERS,
    SMOKING_OPTIONS,
    MEDITATION_OPTIONS,
    SLEEP_COLORS,
    STRESS_COLORS,
    SLEEP_FEAT_IMP,
    STRESS_FEAT_IMP,
    SLEEP_METRICS,
    STRESS_METRICS,
)
from combined_wellness_recommender import get_wellness_recommendations

st.set_page_config(page_title="SleepIQ", page_icon="Moon", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#0e1117; color:#ffffff; }
[data-testid="stHeader"] { background:#0e1117; }
label, .stMarkdown, .stSelectbox label, .stNumberInput label, .stSlider label, .stSelectbox div[data-testid="stWidgetLabel"], .stNumberInput div[data-testid="stWidgetLabel"], .stSlider div[data-testid="stWidgetLabel"] { color:#ffffff !important; }
[data-baseweb="select"] * { color:#111111 !important; }
[data-baseweb="select"] input, [data-baseweb="select"] div, [data-baseweb="popover"] * { color:#111111 !important; }
.card { background:#161b22; border:1px solid #30363d; border-radius:14px; padding:22px 24px; margin-bottom:16px; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px; text-align:center; }
.rec-row { background:#1c2128; border-radius:0 10px 10px 0; padding:14px 18px; margin-bottom:10px; }
.rec-row.quick { border-left:4px solid #3fb950; }
.rec-row.full  { border-left:4px solid #d29922; }
.explain-pos { background:#1c2128; border-left:4px solid #3fb950; border-radius:0 8px 8px 0; padding:10px 16px; margin-bottom:8px; }
.explain-neg { background:#1c2128; border-left:4px solid #e74c3c; border-radius:0 8px 8px 0; padding:10px 16px; margin-bottom:8px; }
.section-title { font-size:13px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#ffffff; margin-bottom:12px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#161b22; border-radius:12px; padding:6px; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px; color:#ffffff; font-weight:600; font-size:15px; padding:10px 24px; border:none; }
.stTabs [aria-selected="true"] { background:#21262d !important; color:#ffffff !important; }
.stButton > button { background:linear-gradient(135deg,#1f6feb,#388bfd); color:white; border:none; border-radius:10px; padding:12px 32px; font-size:16px; font-weight:700; width:100%; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:28px 0 18px 0;">
  <h1 style="font-size:36px;font-weight:900;margin:8px 0 4px 0; color:#ffffff;">SleepIQ</h1>
  <p style="color:#ffffff;font-size:16px;margin:0;">Stress-aware sleep and wellness showcase</p>
</div>
""", unsafe_allow_html=True)


def render_inputs(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 18, 80, 32, 1, key=f"{prefix}_age")
        gender = st.selectbox("Gender", GENDERS, key=f"{prefix}_gender")
        smoking = st.selectbox("Smoking Habit", SMOKING_OPTIONS, key=f"{prefix}_smoking")
        meditation = st.selectbox("Meditation Practice", MEDITATION_OPTIONS, key=f"{prefix}_meditation")
    with c2:
        st.markdown('<div class="section-title">Sleep and Lifestyle</div>', unsafe_allow_html=True)
        sleep_duration = st.slider("Sleep Duration (hrs)", 3.0, 10.0, 6.5, 0.25, key=f"{prefix}_sleep")
        screen_time = st.slider("Screen Time (hrs/day)", 0.0, 12.0, 4.0, 0.5, key=f"{prefix}_screen")
        caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 3, 1, key=f"{prefix}_caffeine")
        alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 1, 1, key=f"{prefix}_alcohol")
    with c3:
        st.markdown('<div class="section-title">Activity and Work</div>', unsafe_allow_html=True)
        physical_activity = st.slider("Exercise Duration (min/day)", 0, 120, 30, 5, key=f"{prefix}_activity")
        daily_steps = st.slider("Daily Step Count", 3000, 10000, 7000, 500, key=f"{prefix}_steps")
        social = st.slider("Social Interactions (hrs/day)", 0.0, 10.0, 2.0, 0.5, key=f"{prefix}_social")
        work_hours = st.slider("Work Hours (per day)", 4.0, 16.0, 8.0, 0.5, key=f"{prefix}_work")
    return {
        "age": age,
        "gender": gender,
        "smoking_habit": smoking,
        "meditation_practice": meditation,
        "sleep_duration": sleep_duration,
        "physical_activity": physical_activity,
        "screen_time": screen_time,
        "caffeine_intake": caffeine,
        "alcohol_intake": alcohol,
        "work_hours": work_hours,
        "social_interactions": social,
        "daily_steps": daily_steps,
    }


def impact_chart(title: str, supporting: list, opposing: list):
    items = ([(r["label"], r["impact"], "#3fb950") for r in supporting[:5]] +
             [(r["label"], r["impact"], "#e74c3c") for r in opposing[:4]])
    if not items:
        return None
    labels, vals, colors = zip(*sorted(items, key=lambda x: x[1]))
    fig = go.Figure(go.Bar(
        x=list(vals),
        y=list(labels),
        orientation="h",
        marker_color=list(colors),
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#ffffff", size=16),
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    fig.update_layout(
        title=title,
        title_font=dict(color="#ffffff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        height=320,
        showlegend=False,
        margin=dict(t=45, b=10, l=10, r=90),
        xaxis=dict(showgrid=False, visible=False, color="#ffffff", tickfont=dict(color="#ffffff")),
        yaxis=dict(autorange="reversed", showgrid=False, color="#ffffff", tickfont=dict(color="#ffffff", size=16)),
    )
    return fig


def showcase_metrics_chart(title: str, cv_metrics: dict, test_metrics: dict, color: str):
    labels = ["Accuracy", "Precision", "Recall", "F1 Macro", "F1 Weighted"]
    cv_values = [
        cv_metrics.get("accuracy_mean", 0.0),
        cv_metrics.get("precision_macro_mean", 0.0),
        cv_metrics.get("recall_macro_mean", 0.0),
        cv_metrics.get("f1_macro_mean", 0.0),
        cv_metrics.get("f1_weighted_mean", 0.0),
    ]
    test_values = [
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("precision_macro", 0.0),
        test_metrics.get("recall_macro", 0.0),
        test_metrics.get("f1_macro", 0.0),
        test_metrics.get("f1_weighted", 0.0),
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="CV", x=labels, y=[v * 100 for v in cv_values], marker_color=color, text=[f"{v*100:.1f}%" for v in cv_values], textposition="outside"))
    fig.add_trace(go.Bar(name="Test", x=labels, y=[v * 100 for v in test_values], marker_color="#8b949e", text=[f"{v*100:.1f}%" for v in test_values], textposition="outside"))
    fig.update_layout(title=title, title_font=dict(color="#ffffff"), barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"), height=360, margin=dict(t=50, b=20, l=20, r=20), yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#30363d", color="#ffffff"), xaxis=dict(showgrid=False, color="#ffffff"), legend=dict(font=dict(color="#ffffff"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def importance_chart(feat_imp):
    top = feat_imp.head(10)
    labels = [
        n.replace("Gender_", "Gender: ")
         .replace("Smoking_Habit_", "Smoking: ")
         .replace("Meditation_Practice_", "Meditation: ")
         .replace("Stress_P_", "Stress P(")
         .replace("_", " ")
        for n in top.index
    ]
    fig = go.Figure(go.Bar(x=top.values * 100, y=labels, orientation="h", marker=dict(color=top.values * 100, colorscale=[[0, "#1f6feb"], [1, "#a371f7"]]), text=[f"{v*100:.1f}%" for v in top.values], textposition="outside"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"), height=340, showlegend=False, margin=dict(t=10, b=10, l=10, r=70), xaxis=dict(showgrid=False, visible=False, color="#ffffff"), yaxis=dict(autorange="reversed", showgrid=False, color="#ffffff"))
    return fig


def combined_importance_chart(stress_imp, sleep_imp):
    def _normalize(series: pd.Series) -> pd.Series:
        cleaned = series.copy()
        cleaned.index = [
            n.replace("Gender_", "Gender: ")
             .replace("Smoking_Habit_", "Smoking: ")
             .replace("Meditation_Practice_", "Meditation: ")
             .replace("Stress_P_", "Stress P(")
             .replace("_", " ")
            for n in cleaned.index
        ]
        cleaned = cleaned.groupby(level=0).sum()
        total = cleaned.sum()
        if total > 0:
            cleaned = cleaned / total
        return cleaned
    merged = pd.concat([_normalize(stress_imp), _normalize(sleep_imp)], axis=1, keys=["Stress", "Sleep"]).fillna(0.0)
    merged["Combined"] = merged["Stress"] + merged["Sleep"]
    top = merged.sort_values("Combined", ascending=False).head(12).sort_values("Combined")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stress Model", x=top["Stress"] * 100, y=top.index.tolist(), orientation="h", marker_color="#3fb950", text=[f"{v*100:.1f}%" if v > 0 else "" for v in top["Stress"]], textposition="outside"))
    fig.add_trace(go.Bar(name="Sleep Model", x=top["Sleep"] * 100, y=top.index.tolist(), orientation="h", marker_color="#388bfd", text=[f"{v*100:.1f}%" if v > 0 else "" for v in top["Sleep"]], textposition="outside"))
    fig.update_layout(title="Combined Feature Importance", title_font=dict(color="#ffffff"), barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"), height=500, margin=dict(t=40, b=10, l=10, r=90), xaxis=dict(title="Relative importance", ticksuffix="%", gridcolor="#30363d", color="#ffffff"), yaxis=dict(showgrid=False, color="#ffffff"), legend=dict(font=dict(color="#ffffff"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def explain_panel(title: str, explanation: dict, color: str) -> None:
    st.markdown(f'<div class="card" style="border-color:{color}44;">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:16px;font-weight:800;color:{color};margin-bottom:8px;">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#ffffff;margin-bottom:14px;font-size:14px;">{explanation["summary"]}</p>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        if explanation["supporting"]:
            st.markdown('<div class="section-title" style="color:#ffffff;">Supporting</div>', unsafe_allow_html=True)
            for row in explanation["supporting"][:5]:
                st.markdown(f'<div class="explain-pos"><b style="color:#ffffff;">{row["label"]}</b></div>', unsafe_allow_html=True)
    with b:
        if explanation["opposing"]:
            st.markdown('<div class="section-title" style="color:#ffffff;">Risk factors</div>', unsafe_allow_html=True)
            for row in explanation["opposing"][:5]:
                st.markdown(f'<div class="explain-neg"><b style="color:#ffffff;">{row["label"]}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def rec_card(rec: dict, css_class: str) -> str:
    direction = "+" if rec["delta"] > 0 else "-"
    color = "#3fb950" if rec["delta"] > 0 else "#f85149"
    return f"""<div class="rec-row {css_class}">
  <span style="font-size:17px;font-weight:800;color:{color};">{direction} {rec['feature']}</span>
  <span style="color:#ffffff;margin:0 10px;">|</span>
  <span style="color:#ffffff;">{rec['from']} to <b>{rec['to']}</b><span style="color:{color};font-size:13px;"> ({direction}{abs(rec['delta'])})</span></span>
  <br><span style="color:#ffffff;font-size:13px;display:block;margin-top:4px;">{rec['tip']}</span>
</div>"""


def score_color(score: float) -> str:
    return "#e74c3c" if score < 5 else "#f39c12" if score <= 7.5 else "#2ecc71"


def display_sleep_score(raw_score: float) -> float:
    score = float(raw_score)
    if score <= 6.5:
        return round(score, 1)
    if score <= 8.0:
        return round(score - 0.2, 1)
    return round(score - 0.35, 1)


def score_label(score: float) -> str:
    score = float(score)
    return "Poor" if score < 5 else "Moderate" if score <= 7.5 else "Good"


FEATURE_TO_INPUT_KEY = {
    "Sleep Duration (hrs)": "sleep_duration",
    "Physical Activity (min/day)": "physical_activity",
    "Screen Time (hrs/day)": "screen_time",
    "Caffeine Intake (cups/day)": "caffeine_intake",
    "Social Interactions (hrs/day)": "social_interactions",
    "Daily Steps": "daily_steps",
}

LONG_TERM_UI_RULES = {
    "Sleep Duration (hrs)": {"step": 0.25, "target": 8.5, "min": 3.0, "max": 10.0},
    "Physical Activity (min/day)": {"step": 5.0, "target": 60.0, "min": 0.0, "max": 120.0},
    "Screen Time (hrs/day)": {"step": 0.5, "target": 2.0, "min": 0.0, "max": 12.0},
    "Caffeine Intake (cups/day)": {"step": 1.0, "target": 1.0, "min": 0.0, "max": 10.0},
    "Social Interactions (hrs/day)": {"step": 0.5, "target": 4.0, "min": 0.0, "max": 10.0},
    "Daily Steps": {"step": 500.0, "target": 9000.0, "min": 3000.0, "max": 10000.0},
}


def apply_recommendations(inputs: dict, recommendations: list) -> dict:
    updated = dict(inputs)
    for rec in recommendations:
        key = FEATURE_TO_INPUT_KEY.get(rec.get("feature"))
        if key is not None:
            updated[key] = rec.get("to")
    return updated


def _rebuild_recommendations(original_inputs: dict, working_inputs: dict, recommendations: list) -> list:
    rebuilt = []
    for rec in recommendations:
        feature = rec.get("feature")
        key = FEATURE_TO_INPUT_KEY.get(feature)
        if key is None:
            rebuilt.append(rec)
            continue
        original = float(original_inputs[key])
        updated = float(working_inputs[key])
        rebuilt.append({**rec, "from": original, "to": updated, "delta": round(updated - original, 2)})
    return rebuilt


def refine_long_term_plan(original_inputs: dict, full_fix: dict, threshold: float = 7.5) -> tuple[dict, float]:
    working_inputs = apply_recommendations(original_inputs, full_fix.get("recommendations", []))
    working_score = display_sleep_score(get_wellness_recommendations(**working_inputs)["current_sleep_score"])
    selected = [rec for rec in full_fix.get("recommendations", []) if rec.get("feature") in LONG_TERM_UI_RULES]
    if not selected or working_score >= threshold:
        return full_fix, working_score

    best_inputs = dict(working_inputs)
    best_score = working_score

    for _ in range(18):
        if best_score >= threshold:
            break
        improved = False
        candidate_inputs = None
        candidate_score = best_score
        candidate_cost = float('inf')

        for rec in selected:
            feature = rec["feature"]
            key = FEATURE_TO_INPUT_KEY[feature]
            rules = LONG_TERM_UI_RULES[feature]
            current = float(best_inputs[key])
            target = float(rules["target"])
            step = float(rules["step"])

            if abs(current - target) < 1e-9:
                continue

            next_value = min(target, current + step) if target > current else max(target, current - step)
            next_value = min(float(rules["max"]), max(float(rules["min"]), next_value))
            if abs(next_value - current) < 1e-9:
                continue

            trial_inputs = dict(best_inputs)
            trial_inputs[key] = next_value
            trial_score = display_sleep_score(get_wellness_recommendations(**trial_inputs)["current_sleep_score"])
            trial_cost = abs(next_value - float(original_inputs[key])) / max(step, 1.0)

            if trial_score >= threshold:
                if (candidate_score < threshold) or (trial_cost < candidate_cost) or (abs(trial_cost - candidate_cost) <= 1e-9 and trial_score > candidate_score):
                    candidate_inputs = trial_inputs
                    candidate_score = trial_score
                    candidate_cost = trial_cost
                    improved = True
            elif trial_score > candidate_score + 1e-9:
                candidate_inputs = trial_inputs
                candidate_score = trial_score
                candidate_cost = trial_cost
                improved = True
            elif abs(trial_score - candidate_score) <= 1e-9 and trial_cost < candidate_cost:
                candidate_inputs = trial_inputs
                candidate_score = trial_score
                candidate_cost = trial_cost
                improved = True

        if not improved or candidate_inputs is None:
            break

        best_inputs = candidate_inputs
        best_score = candidate_score

    updated_fix = dict(full_fix)
    updated_fix["recommendations"] = _rebuild_recommendations(original_inputs, best_inputs, selected)
    return updated_fix, best_score



def stress_rank(label: str) -> int:
    return {"High": 0, "Medium": 1, "Low": 2}.get(label, -1)


def pick_better_result(primary: dict, primary_score: float, secondary: dict, secondary_score: float) -> tuple[dict, float]:
    primary_stress = stress_rank(primary.get("new_stress", {}).get("label", ""))
    secondary_stress = stress_rank(secondary.get("new_stress", {}).get("label", ""))
    if secondary_score > primary_score + 1e-9:
        return secondary, secondary_score
    if abs(secondary_score - primary_score) <= 1e-9 and secondary_stress > primary_stress:
        return secondary, secondary_score
    return primary, primary_score


def outcome_card(title: str, current_score: float, new_score: float, current_stress: str, new_stress: str, accent: str) -> str:
    current_sleep_label = score_label(current_score)
    new_sleep_label = score_label(new_score)
    current_color = score_color(current_score)
    new_color = score_color(new_score)
    return f"""<div class="card" style="border-color:{accent}; margin-bottom:14px; background:linear-gradient(180deg, #161b22 0%, #141922 100%);">
  <div style="font-size:16px;font-weight:800;color:{accent};margin-bottom:14px;">{title}</div>
  <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:180px;background:#11161d;border:1px solid #30363d;border-radius:12px;padding:14px 16px;">
      <div style="color:#ffffff;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">Sleep Score</div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-top:10px;">
        <span style="font-size:28px;font-weight:900;color:{current_color};">{current_score:.1f}</span>
        <span style="color:#ffffff;font-size:18px;">to</span>
        <span style="font-size:28px;font-weight:900;color:{new_color};">{new_score:.1f}</span>
      </div>
      <div style="margin-top:8px;color:#ffffff;font-size:15px;">{current_sleep_label} to <span style="color:{new_color};font-weight:800;">{new_sleep_label}</span></div>
    </div>
    <div style="flex:1;min-width:180px;background:#11161d;border:1px solid #30363d;border-radius:12px;padding:14px 16px;">
      <div style="color:#ffffff;font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">Stress</div>
      <div style="margin-top:10px;font-size:24px;font-weight:900;">
        <span style="color:{STRESS_COLORS.get(current_stress, '#ffffff')};">{current_stress}</span>
        <span style="color:#ffffff;font-size:18px;font-weight:700;margin:0 10px;">to</span>
        <span style="color:{STRESS_COLORS.get(new_stress, '#ffffff')};">{new_stress}</span>
      </div>
    </div>
  </div>
</div>"""


tab1, tab2, tab3 = st.tabs(["Predict", "Optimise", "About"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    inputs = render_inputs("predict")
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Analyse My Sleep and Stress", key="predict_btn"):
        results = predict_both(**inputs)
        stress_exp = explain_stress(**inputs)
        sleep_exp = explain_sleep(**inputs)
        wellness = get_wellness_recommendations(**inputs)
        current_sleep = display_sleep_score(wellness["current_sleep_score"])
        sleep_label = score_label(current_sleep)
        sleep_color = score_color(current_sleep)
        explain_panel(f"Stress Level: {results['stress']['label']} - key drivers", stress_exp, results['stress']['color'])
        explain_panel(f"Sleep Quality: {sleep_label} - key drivers", sleep_exp, sleep_color)
        c1, c2 = st.columns(2)
        with c1:
            fig = impact_chart("Stress Factor Impact", stress_exp["supporting"], stress_exp["opposing"])
            if fig:
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with c2:
            fig = impact_chart("Sleep Factor Impact", sleep_exp["supporting"], sleep_exp["opposing"])
            if fig:
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        score_col, stress_col = st.columns(2)
        with score_col:
            color = score_color(current_sleep)
            label = score_label(current_sleep)
            st.markdown(f"""<div class="metric-card"><div style="color:#ffffff;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Sleep Score</div><div style="font-size:56px;font-weight:900;color:{color};margin:8px 0;">{current_sleep:.1f}</div><div style="color:{color};font-size:18px;font-weight:800;margin-bottom:6px;">{label}</div><div style="color:#ffffff;font-size:14px;">out of 10</div></div>""", unsafe_allow_html=True)
        with stress_col:
            stress_label = results["stress"]["label"]
            st.markdown(f"""<div class="metric-card"><div style="color:#ffffff;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Stress Result</div><div style="font-size:44px;font-weight:900;color:{STRESS_COLORS[stress_label]};margin:14px 0 10px 0;">{stress_label}</div><div style="color:#ffffff;font-size:14px;">calibrated stress output</div></div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    opt_inputs = render_inputs("opt")
    target = st.selectbox("Sleep quality target", ["Good", "Moderate"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Find My Optimisations", key="opt_btn"):
        result = optimize(sleep_target=target, **opt_inputs)
        qw = result["quick_wins"]
        ff = result["full_fix"]
        current_wellness = get_wellness_recommendations(**opt_inputs)
        qw_inputs = apply_recommendations(opt_inputs, qw["recommendations"])
        ff_inputs = apply_recommendations(opt_inputs, ff["recommendations"])
        qw_wellness = get_wellness_recommendations(**qw_inputs)
        ff_wellness = get_wellness_recommendations(**ff_inputs)
        current_score = display_sleep_score(current_wellness["current_sleep_score"])
        qw_score = display_sleep_score(qw_wellness["current_sleep_score"])
        ff_score = display_sleep_score(ff_wellness["current_sleep_score"])
        ff, ff_score = refine_long_term_plan(opt_inputs, ff, threshold=7.5)
        if ff_score < 7.5:
            ff, ff_score = pick_better_result(ff, ff_score, qw, qw_score)
        left, right = st.columns(2)
        with left:
            st.markdown(outcome_card(
                "Quick Wins Outcome",
                current_score,
                qw_score,
                result["base_stress"]["label"],
                qw["new_stress"]["label"],
                "#3fb950",
            ), unsafe_allow_html=True)
            st.markdown('<div style="font-size:18px;font-weight:800;color:#3fb950;margin:8px 0 12px 0;">Quick Wins</div>', unsafe_allow_html=True)
            for rec in qw["recommendations"]:
                st.markdown(rec_card(rec, "quick"), unsafe_allow_html=True)
        with right:
            st.markdown(outcome_card(
                "Strongest Fixes Outcome",
                current_score,
                ff_score,
                result["base_stress"]["label"],
                ff["new_stress"]["label"],
                "#d29922",
            ), unsafe_allow_html=True)
            st.markdown('<div style="font-size:18px;font-weight:800;color:#d29922;margin:8px 0 12px 0;">Strongest Fixes</div>', unsafe_allow_html=True)
            for rec in ff["recommendations"]:
                st.markdown(rec_card(rec, "full"), unsafe_allow_html=True)

with tab3:
    stress_cv = STRESS_METRICS.get("cv_classification", {})
    stress_test = STRESS_METRICS.get("test_classification", {})
    sleep_cv = SLEEP_METRICS.get("cv_classification", {})
    sleep_test = SLEEP_METRICS.get("test_classification", {})
    st.markdown('<div class="card"><h3 style="color:#ffffff;margin-top:0;">About</h3><p style="color:#ffffff;line-height:1.7;font-size:14px;">This showcase predicts stress first, then uses that calibrated stress signal to estimate sleep quality and a combined sleep score.</p></div>', unsafe_allow_html=True)
    st.plotly_chart(showcase_metrics_chart("Stress Showcase Metrics", stress_cv, stress_test, "#3fb950"), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(showcase_metrics_chart("Sleep Showcase Metrics", sleep_cv, sleep_test, "#388bfd"), width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="card"><h3 style="color:#ffffff;margin-top:0;">Disclaimer</h3><p style="color:#ffffff;line-height:1.7;font-size:14px;">This tool is for educational and demonstration purposes only and is not medical advice.</p></div>', unsafe_allow_html=True)
