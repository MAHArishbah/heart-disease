import os
import requests
import streamlit as st
import math
import altair as alt
import pandas as pd

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

LABELS = {"age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting BP",
          "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
          "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression"}



st.set_page_config(page_title="Heart disease risk", layout="centered")
st.title("Heart disease risk")
st.caption("Classical ML model on the pooled UCI data. Not a medical device.")

with st.form("patient"):
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
    st.caption("Fields outside their range are marked \u24d8 and block scoring until fixed.")
    submitted = st.form_submit_button("Score", type="primary")

if submitted:

    patient = {
        "age": age, "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
        "trestbps": optional(trestbps, no_bp),
        "chol": optional(chol, no_chol),
        "thalach": optional(thalach, no_stress),
        "exang": optional(exang, no_stress),
        "oldpeak": optional(oldpeak, no_stress),
        # "slope": None if no_stress else slope,   # slope comes from the exercise test, so no test means no slope
        # "ca": ca,
        # "thal": thal,
    }
    try:
        r = requests.post(f"{API_URL}/explain", json=patient, headers={**auth_headers(),"x-client":"ui"}, timeout=10)
    except requests.RequestException as exc:          # network problem: no response at all
        st.error(f"Could not reach the model service: {exc}")
        st.stop()

    if r.status_code == 422:                          # the API rejected the input: say which field and why
        for err in r.json().get("detail", []):
            field = ".".join(map(str, err.get("loc", [])[1:])) or "input"
            st.error(f"{field}: {err.get('msg')}")
        st.stop()
    if not r.ok:                                      # anything else (403, 500, 503)
        st.error(f"Model service returned {r.status_code}: {r.text[:200]}")
        st.stop()

    result = r.json()
    st.metric("Probability of disease", f"{result['probability']:.1%}")
    st.progress(min(result["probability"], 1.0))
    st.write(f"**{result['label']}** at the operating threshold {result['threshold']:.2f} "
             f"(model `{r.headers.get('x-model-version', '?')}`).")
    st.caption("The threshold comes from a 4:1 false-negative to false-positive cost "
               "ratio, not the default 0.5.")
    st.subheader("What drove this score")
    base_p = 1 / (1 + math.exp(-result["base_value"]))
    contrib = pd.DataFrame(
        [(LABELS.get(k, k), v, "not provided" if patient.get(k) is None else "provided")
         for k, v in result["contributions"].items()],
        columns=["feature", "contribution", "input"])
    contrib["effect"] = contrib["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)

    bars = alt.Chart(contrib).mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("contribution:Q", title="Contribution to risk (log-odds)"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.Color("effect:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#e34948", "#2a78d6"])),
        tooltip=["feature", "input", alt.Tooltip("contribution:Q", format="+.3f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    st.altair_chart(zero + bars,width="stretch")
    st.caption(f"Bars show how far each input moved this patient away from the average training "
               f"patient ({base_p:.0%}). Contributions reflect associations in the training data "
               f"(79% male, 1980s cohorts), not causes. Missing inputs are filled with the training "
               f"median/mode and can still contribute.")


    with st.expander("Values sent to the model"):
        st.json(patient)
