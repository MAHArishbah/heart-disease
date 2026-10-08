# File: ui/streamlit_app.py · new
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")


def auth_headers() -> dict:
    """Nothing locally; a Cloud Run identity token when calling the deployed API."""
    if not API_URL.startswith("https://"):
        return {}
    import google.auth.transport.requests
    import google.oauth2.id_token
    token = google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), API_URL)
    return {"Authorization": f"Bearer {token}"}


def optional(value, missing: bool):
    return None if missing else value


st.set_page_config(page_title="Heart disease risk", layout="centered")
st.title("Heart disease risk")
st.caption("Elastic-net logistic regression on the pooled UCI data. Not a medical device.")

left, right = st.columns(2)
with left:
    age = st.number_input("Age (years)", 18, 100, 55)
    sex = st.radio("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female", horizontal=True)
    cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda v: {
        1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal", 4: "Asymptomatic"}[v])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
    no_bp = st.checkbox("BP not measured")
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    no_chol = st.checkbox("Cholesterol not measured")
with right:
    fbs = st.radio("Fasting blood sugar > 120", [0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True)
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda v: {
        0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[v])
    thalach = st.number_input("Max heart rate", 60, 220, 150)
    exang = st.radio("Exercise angina", [0, 1], format_func=lambda v: "Yes" if v else "No", horizontal=True)
    oldpeak = st.number_input("ST depression", -3.0, 7.0, 1.0, step=0.1)
    no_stress = st.checkbox("No exercise test done")

if st.button("Score", type="primary"):
    patient = {
        "age": age, "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
        "trestbps": optional(trestbps, no_bp),
        "chol": optional(chol, no_chol),
        "thalach": optional(thalach, no_stress),
        "exang": optional(exang, no_stress),
        "oldpeak": optional(oldpeak, no_stress),
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=patient, headers=auth_headers(), timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the model service: {exc}")
    else:
        result = r.json()
        st.metric("Probability of disease", f"{result['probability']:.1%}")
        st.progress(min(result["probability"], 1.0))
        st.write(f"**{result['label']}** at the operating threshold {result['threshold']:.2f} "
                 f"(model `{r.headers.get('x-model-version', '?')}`).")
        st.caption("The threshold comes from a 4:1 false-negative to false-positive cost "
                   "ratio, not the default 0.5.")