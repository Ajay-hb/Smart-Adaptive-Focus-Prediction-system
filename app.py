import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
ROWS = 500


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


def main():
    st.set_page_config(page_title="Smart Adaptive Focus Prediction", page_icon="🎯", layout="wide")
    st.title("🎯 Smart Adaptive Focus Prediction System")
    st.caption("Streamlit implementation of your notebook workflow using XGBoost.")

    df = generate_dataset()
    model, feature_columns, feature_dtypes, categorical_cols, label_encoders = train_model(df)

    with st.expander("Preview generated dataset"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    st.subheader("Enter your current daily factors")
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

        st.metric("Predicted Focus Score", f"{clipped_prediction:.2f}/100")

        if clipped_prediction >= 70:
            st.success("Great! Your current setup is likely to support strong focus.")
        elif clipped_prediction >= 45:
            st.info("Moderate focus predicted. A few adjustments could improve it.")
        else:
            st.warning("Lower focus predicted. Suggestions below can help improve it.")

        st.subheader("Improvement suggestions")
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
                st.write(f"- {text}. Estimated score: **{improved_score:.2f}**")
        else:
            st.write("- No single change from the suggestion list improved your current score.")


if __name__ == "__main__":
    main()
