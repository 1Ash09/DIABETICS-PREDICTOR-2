from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Diabetes Risk Assessor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .block-container {max-width: 1100px; padding-top: 2.5rem; padding-bottom: 3rem;}
        .hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(70,130,180,.10), rgba(128,128,128,.06));
            margin-bottom: 1.4rem;
        }
        .hero h1 {margin: 0 0 .35rem 0; font-size: 2.2rem;}
        .hero p {margin: 0; opacity: .75; font-size: 1rem;}
        .status {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            border: 1px solid rgba(128,128,128,.25);
            font-size: .82rem;
            margin-top: .9rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.20);
            border-radius: 14px;
            padding: .8rem;
        }
        .disclaimer {
            margin-top: 2rem;
            padding: .9rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,.18);
            font-size: .85rem;
            opacity: .78;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Diabetes Risk Assessor</h1>
        <p>Enter the patient information below to generate a prediction from the trained machine-learning model.</p>
        <div class="status">● Model ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

TARGET_SOURCE = "glyhb"
TARGET = "Outcome"
THRESHOLD = 0.45

DROP_COLUMNS = [
    "id", "glyhb", "bp.2s", "bp.2d", "hip", "waist",
    "height", "weight", "frame", "location"
]

DEFAULTS = {
    "chol": 220.0,
    "stab.glu": 110.0,
    "hdl": 45.0,
    "ratio": 4.8,
    "age": 52,
    "bp.1s": 140.0,
    "bp.1d": 88.0,
    "time.ppn": 300.0,
    "gender": "male",
}


@st.cache_resource
def train_model():
    data_path = Path(__file__).resolve().parent / "diabetes.csv"
    if not data_path.exists():
        raise FileNotFoundError("diabetes.csv was not found in the app folder.")

    raw = pd.read_csv(data_path)

    if TARGET_SOURCE not in raw.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_SOURCE}' column.")

    df = raw.dropna(subset=[TARGET_SOURCE]).copy()
    df[TARGET] = (df[TARGET_SOURCE] >= 6.5).astype(int)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].dropna().empty:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("The dataset must contain both outcome classes.")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    return model, scaler, X.columns.tolist()


def make_patient(values, columns):
    row = pd.DataFrame([values])
    for col in columns:
        if col not in row.columns:
            row[col] = 0
    return row[columns]


def risk_label(probability):
    if probability >= 0.70:
        return "High"
    if probability >= THRESHOLD:
        return "Elevated"
    return "Lower"


try:
    model, scaler, columns = train_model()
except Exception as exc:
    st.error(f"Could not load the diabetes model: {exc}")
    st.stop()

st.subheader("Patient information")
st.caption("Enter the values used by the model. All fields below are required for a prediction.")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        chol = st.number_input("Cholesterol", min_value=0.0, value=DEFAULTS["chol"], step=1.0)
        stab_glu = st.number_input("Stable glucose", min_value=0.0, value=DEFAULTS["stab.glu"], step=1.0)
        hdl = st.number_input("HDL", min_value=0.0, value=DEFAULTS["hdl"], step=1.0)

    with c2:
        ratio = st.number_input("Cholesterol / HDL ratio", min_value=0.0, value=DEFAULTS["ratio"], step=0.1)
        age = st.number_input("Age", min_value=1, max_value=120, value=DEFAULTS["age"])
        gender = st.selectbox("Gender", ["female", "male"], index=1)

    with c3:
        bp_1s = st.number_input("Systolic BP", min_value=0.0, value=DEFAULTS["bp.1s"], step=1.0)
        bp_1d = st.number_input("Diastolic BP", min_value=0.0, value=DEFAULTS["bp.1d"], step=1.0)
        time_ppn = st.number_input("Time since previous meal", min_value=0.0, value=DEFAULTS["time.ppn"], step=1.0)

    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
    assess = st.button("Assess diabetes risk", type="primary", use_container_width=True)

if assess:
    patient_values = {
        "chol": chol,
        "stab.glu": stab_glu,
        "hdl": hdl,
        "ratio": ratio,
        "age": age,
        "bp.1s": bp_1s,
        "bp.1d": bp_1d,
        "time.ppn": time_ppn,
        "gender_male": 1 if gender == "male" else 0,
    }

    try:
        patient_df = make_patient(patient_values, columns)
        scaled_patient = scaler.transform(patient_df)
        probability = float(model.predict_proba(scaled_patient)[0][1])
        prediction = int(probability >= THRESHOLD)

        st.divider()
        st.subheader("Prediction result")

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Estimated probability", f"{probability * 100:.2f}%")
        with r2:
            st.metric("Risk category", risk_label(probability))

        st.progress(min(probability, 1.0), text=f"Estimated probability: {probability * 100:.1f}%")

        if prediction == 1:
            st.error("Model prediction: DIABETIC (1)")
        else:
            st.success("Model prediction: NON-DIABETIC (0)")

        st.caption("The prediction uses the 45% probability cutoff from the supplied notebook.")
    except Exception as exc:
        st.error(f"Prediction error: {exc}")

st.markdown(
    """
    <div class="disclaimer">
    <strong>Important:</strong> This application provides a machine-learning prediction based on the supplied dataset and notebook. It is not a medical diagnosis and should not be used as a substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)
