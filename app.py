from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
ROWS = 500


st.set_page_config(
    page_title="Smart Adaptive Focus Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def categorize_sleep(hours: int) -> str:
    if hours <= 5:
        return "Low"
    if hours <= 7:
        return "Medium"
    return "High"


@st.cache_data
def generate_dataset(rows: int = ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    np.random.seed(seed)
    data = {
        "sleep_hours": np.random.randint(3, 10, rows),
        "screen_time": np.random.randint(1, 10, rows),
        "noise_level": np.random.randint(1, 4, rows),
        "time_of_day": np.random.randint(1, 4, rows),
        "caffeine": np.random.randint(0, 4, rows),
        "stress_level": np.random.randint(1, 6, rows),
        "exercise": np.random.randint(0, 4, rows),
        "mood": np.random.randint(1, 6, rows),
        "task_difficulty": np.random.randint(1, 4, rows),
    }
    df = pd.DataFrame(data)
    df["focus_score"] = (
        df["sleep_hours"] * 10
        - df["screen_time"] * 4
        - df["noise_level"] * 5
        + df["caffeine"] * 3
        - df["stress_level"] * 3
        + df["exercise"] * 4
        + df["mood"] * 2
        - df["task_difficulty"] * 2
        + np.random.randint(-10, 10, rows)
    )
    df["focus_score"] = df["focus_score"].clip(0, 100)
    return df


def preprocess_training_data(df: pd.DataFrame):
    df_processed = df.copy()
    df_processed["sleep_category"] = df_processed["sleep_hours"].apply(categorize_sleep).astype("category")
    df_processed["noise_level"] = df_processed["noise_level"].astype("category")
    df_processed["time_of_day"] = df_processed["time_of_day"].astype("category")
    df_processed["sleep_stress_interaction"] = df_processed["sleep_hours"] * df_processed["stress_level"]
    df_processed["caffeine_sleep_interaction"] = df_processed["caffeine"] * df_processed["sleep_hours"]

    categorical_cols = ["noise_level", "time_of_day", "sleep_category"]

    label_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_processed[col] = encoder.fit_transform(df_processed[col])
        label_encoders[col] = encoder

    X = df_processed.drop("focus_score", axis=1)
    y = df_processed["focus_score"]
    return X, y, categorical_cols, label_encoders


@st.cache_resource
def train_model(df: pd.DataFrame):
    X, y, categorical_cols, label_encoders = preprocess_training_data(df)
    model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
    )
    model.fit(X, y)
    return model, X.columns.tolist(), X.dtypes.to_dict(), categorical_cols, label_encoders


def prepare_single_input(
    input_dict: dict,
    feature_columns: List[str],
    feature_dtypes: dict,
    categorical_cols: List[str],
    label_encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    sample = pd.DataFrame([input_dict])
    sample["sleep_category"] = sample["sleep_hours"].apply(categorize_sleep).astype("category")
    sample["noise_level"] = sample["noise_level"].astype("category")
    sample["time_of_day"] = sample["time_of_day"].astype("category")
    sample["sleep_stress_interaction"] = sample["sleep_hours"] * sample["stress_level"]
    sample["caffeine_sleep_interaction"] = sample["caffeine"] * sample["sleep_hours"]

    for col in categorical_cols:
        sample[col] = label_encoders[col].transform(sample[col])

    aligned = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        aligned[col] = sample[col] if col in sample.columns else 0

    aligned = aligned.astype(feature_dtypes)
    return aligned


def predict_focus(
    model: XGBRegressor,
    input_dict: dict,
    feature_columns: List[str],
    feature_dtypes: dict,
    categorical_cols: List[str],
    label_encoders: Dict[str, LabelEncoder],
) -> float:
    X_single = prepare_single_input(
        input_dict,
        feature_columns,
        feature_dtypes,
        categorical_cols,
        label_encoders,
    )
    return float(model.predict(X_single)[0])


def generate_suggestions(
    current_input: dict,
    baseline_score: float,
    model: XGBRegressor,
    feature_columns: List[str],
    feature_dtypes: dict,
    categorical_cols: List[str],
    label_encoders: Dict[str, LabelEncoder],
) -> List[Tuple[str, float]]:
    candidates = [
        ("Increase sleep hours to 9", "sleep_hours", 9),
        ("Reduce screen time to 6", "screen_time", 6),
        ("Reduce stress level to 1", "stress_level", 1),
        ("Increase exercise to 3", "exercise", 3),
        ("Improve mood to 5", "mood", 5),
    ]

    suggestions = []
    for message, key, value in candidates:
        modified = dict(current_input)
        modified[key] = value
        score = predict_focus(
            model,
            modified,
            feature_columns,
            feature_dtypes,
            categorical_cols,
            label_encoders,
        )
        if score > baseline_score:
            suggestions.append((message, score))

    suggestions.sort(key=lambda item: item[1], reverse=True)
    return suggestions


def score_status(score: float) -> Tuple[str, str, str]:
    if score >= 70:
        return "Strong Focus", "#2f7d5c", "Your current setup is likely to support deep work."
    if score >= 45:
        return "Moderate Focus", "#b26a21", "A few focused adjustments could improve your output."
    return "Needs Recovery", "#b53d47", "Your current setup may make sustained concentration harder."


st.markdown(
    """
<style>
:root {
    --paper: #f6f3ee;
    --surface: #ffffff;
    --ink: #20232a;
    --muted: #68707d;
    --line: #ded8cf;
    --plum: #5e4b8b;
    --sage: #5f8d73;
    --clay: #c46b4d;
    --gold: #d9a441;
    --soft-plum: rgba(94, 75, 139, 0.12);
    --soft-sage: rgba(95, 141, 115, 0.14);
}

html, body, [class*="css"] {
    font-family: Georgia, "Times New Roman", serif;
}

.stApp {
    background:
        linear-gradient(120deg, rgba(94,75,139,0.08), transparent 28%),
        linear-gradient(240deg, rgba(196,107,77,0.10), transparent 30%),
        var(--paper);
    color: var(--ink);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #efe8dc;
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
    color: var(--ink);
}

.hero {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 34px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.68)),
        linear-gradient(110deg, rgba(94,75,139,0.12), rgba(95,141,115,0.10));
    box-shadow: 0 24px 70px rgba(64, 48, 34, 0.12);
    position: relative;
    overflow: hidden;
    animation: riseIn 650ms ease both;
}

.hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(32,35,42,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32,35,42,0.035) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: linear-gradient(90deg, transparent, black 12%, black 80%, transparent);
}

.hero-inner {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(240px, .7fr);
    gap: 28px;
    align-items: center;
}

.eyebrow {
    color: var(--plum);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

.hero h1 {
    margin: 0;
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.04;
    letter-spacing: 0;
    color: var(--ink);
}

.hero p {
    margin: 17px 0 0;
    max-width: 680px;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.7;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

.hero-badge {
    border-radius: 8px;
    padding: 18px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.72);
}

.hero-badge strong {
    display: block;
    font-size: 34px;
    color: var(--sage);
    line-height: 1;
}

.hero-badge span {
    display: block;
    margin-top: 8px;
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 13px;
}

.section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin: 24px 0 12px;
}

.section-title h2 {
    margin: 0;
    font-size: 24px;
    color: var(--ink);
}

.pill {
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.70);
    padding: 7px 12px;
    color: var(--muted);
    font-size: 12px;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-weight: 800;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 22px;
    background: rgba(255,255,255,0.82);
    box-shadow: 0 18px 50px rgba(64,48,34,0.10);
    animation: riseIn 700ms ease both;
}

.stat-card {
    min-height: 110px;
    border-radius: 8px;
    border: 1px solid var(--line);
    padding: 17px;
    background: var(--surface);
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(94,75,139,0.34);
    box-shadow: 0 18px 34px rgba(64,48,34,0.13);
}

.stat-card span {
    display: block;
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.stat-card strong {
    display: block;
    margin-top: 10px;
    font-size: 28px;
    line-height: 1.05;
    color: var(--ink);
}

.result-panel {
    border: 1px solid rgba(95,141,115,0.32);
    border-radius: 8px;
    padding: 24px;
    background:
        linear-gradient(135deg, rgba(95,141,115,0.12), rgba(217,164,65,0.11)),
        #fffdf9;
    box-shadow: 0 24px 64px rgba(64,48,34,0.13);
    animation: popIn 520ms ease both;
}

.score {
    font-size: clamp(40px, 6vw, 68px);
    font-weight: 900;
    line-height: 1;
    color: var(--score-color);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

.meter {
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    background: #e6ded3;
    margin: 18px 0 12px;
}

.meter span {
    display: block;
    height: 100%;
    width: var(--score-width);
    background: var(--score-color);
    border-radius: inherit;
    animation: grow 820ms cubic-bezier(.2,.9,.2,1) both;
}

.suggestion {
    border: 1px solid var(--line);
    border-left: 5px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    background: #fffdf9;
    margin-bottom: 10px;
    box-shadow: 0 10px 24px rgba(64,48,34,0.07);
}

.suggestion strong {
    color: var(--ink);
}

.empty-note {
    border: 1px dashed var(--line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(255,255,255,0.60);
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(255,255,255,0.82);
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

.stButton > button {
    min-height: 50px;
    width: 100%;
    border-radius: 8px;
    border: 1px solid rgba(94,75,139,0.30);
    background: linear-gradient(90deg, var(--plum), var(--clay));
    color: white;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-weight: 850;
    box-shadow: 0 16px 34px rgba(94,75,139,0.18);
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 42px rgba(94,75,139,0.24);
}

label, .stSlider label, .stSelectbox label {
    color: var(--ink) !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-weight: 750 !important;
}

div[data-baseweb="select"] > div {
    background: #fffdf9 !important;
    border-color: var(--line) !important;
    border-radius: 8px !important;
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(.985) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes grow {
    from { width: 0; }
    to { width: var(--score-width); }
}

@media (max-width: 860px) {
    .hero-inner {
        grid-template-columns: 1fr;
    }
    .hero {
        padding: 24px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


df = generate_dataset()
model, feature_columns, feature_dtypes, categorical_cols, label_encoders = train_model(df)


st.markdown(
    """
<div class="hero">
    <div class="hero-inner">
        <div>
            <div class="eyebrow">Adaptive focus analytics</div>
            <h1>Smart Focus Prediction System</h1>
            <p>Estimate your focus readiness from sleep, stress, environment, mood, and task conditions with a clean XGBoost-powered workflow.</p>
        </div>
        <div class="hero-badge">
            <strong>0-100</strong>
            <span>Focus score with practical improvement recommendations.</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Model Console")
    st.caption("Synthetic dataset trained inside the app session.")
    st.divider()
    st.metric("Training Rows", f"{len(df):,}")
    st.metric("Features", len(feature_columns))
    st.metric("Seed", RANDOM_STATE)
    st.divider()
    with st.expander("Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)


st.markdown(
    """
<div class="section-title">
    <h2>Daily Factors</h2>
    <span class="pill">Current state</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    sleep_hours = st.slider("Sleep hours", 3, 9, 8)
    screen_time = st.slider("Screen time (hours)", 1, 9, 6)
    noise_level = st.selectbox("Noise level", [1, 2, 3], index=1)

with c2:
    time_of_day = st.selectbox("Time of day", [1, 2, 3], index=0)
    caffeine = st.slider("Caffeine (0-3)", 0, 3, 1)
    stress_level = st.slider("Stress level (1-5)", 1, 5, 3)

with c3:
    exercise = st.slider("Exercise (0-3)", 0, 3, 2)
    mood = st.slider("Mood (1-5)", 1, 5, 4)
    task_difficulty = st.selectbox("Task difficulty", [1, 2, 3], index=1)

st.markdown("</div>", unsafe_allow_html=True)


summary_cols = st.columns(4)
summary_items = [
    ("Sleep", f"{sleep_hours}h"),
    ("Screen", f"{screen_time}h"),
    ("Stress", f"{stress_level}/5"),
    ("Mood", f"{mood}/5"),
]

for col, (label, value) in zip(summary_cols, summary_items):
    with col:
        st.markdown(
            f"""
<div class="stat-card">
    <span>{label}</span>
    <strong>{value}</strong>
</div>
""",
            unsafe_allow_html=True,
        )


user_input = {
    "sleep_hours": sleep_hours,
    "screen_time": screen_time,
    "noise_level": noise_level,
    "time_of_day": time_of_day,
    "caffeine": caffeine,
    "stress_level": stress_level,
    "exercise": exercise,
    "mood": mood,
    "task_difficulty": task_difficulty,
}


st.markdown(
    """
<div class="section-title">
    <h2>Focus Forecast</h2>
    <span class="pill">Prediction</span>
</div>
""",
    unsafe_allow_html=True,
)

if st.button("Predict Focus Score", type="primary"):
    prediction = predict_focus(
        model,
        user_input,
        feature_columns,
        feature_dtypes,
        categorical_cols,
        label_encoders,
    )
    clipped_prediction = float(np.clip(prediction, 0, 100))
    status, status_color, status_text = score_status(clipped_prediction)

    result_col, details_col = st.columns([1.35, 0.75])

    with result_col:
        st.markdown(
            f"""
<div class="result-panel" style="--score-color:{status_color}; --score-width:{clipped_prediction:.0f}%;">
    <div class="eyebrow">Predicted Focus Score</div>
    <div class="score">{clipped_prediction:.2f}/100</div>
    <div class="meter"><span></span></div>
    <h3 style="margin:12px 0 4px;color:{status_color};">{status}</h3>
    <p style="margin:0;color:#68707d;font-family:Inter,ui-sans-serif,system-ui,sans-serif;">{status_text}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with details_col:
        st.metric("Sleep Category", categorize_sleep(sleep_hours))
        st.metric("Environment Load", noise_level + task_difficulty)
        st.metric("Recovery Signals", exercise + mood)

    st.markdown(
        """
<div class="section-title">
    <h2>Improvement Suggestions</h2>
    <span class="pill">Best single changes</span>
</div>
""",
        unsafe_allow_html=True,
    )

    suggestions = generate_suggestions(
        user_input,
        clipped_prediction,
        model,
        feature_columns,
        feature_dtypes,
        categorical_cols,
        label_encoders,
    )

    if suggestions:
        for text, improved_score in suggestions[:4]:
            delta = improved_score - clipped_prediction
            st.markdown(
                f"""
<div class="suggestion" style="--accent:{status_color};">
    <strong>{text}</strong><br>
    Estimated score: {improved_score:.2f} <span style="color:#5f8d73;">(+{delta:.2f})</span>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
<div class="empty-note">
    No single change from the suggestion list improved your current score.
</div>
""",
            unsafe_allow_html=True,
        )
