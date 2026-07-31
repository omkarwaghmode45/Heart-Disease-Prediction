import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/random_forest.pkl")

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Prediction System")
st.write(
    "Predict the likelihood of heart disease using patient clinical information."
)

st.header("Patient Information")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [True, False]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "lv hypertrophy",
            "st-t abnormality"
        ]
    )

    thalch = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [True, False]
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=-5.0,
        max_value=10.0,
        value=1.0
    )

# ---------------- PREDICTION ----------------

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Heart Disease"):

    sex_male = 1 if sex == "Male" else 0

    fbs = int(fbs)
    exang = int(exang)

    # Chest Pain Encoding
    cp_non_anginal = 0
    cp_typical_angina = 0

    if cp == "non-anginal":
        cp_non_anginal = 1
    elif cp == "typical angina":
        cp_typical_angina = 1

    # Rest ECG Encoding
    restecg_normal = 0
    restecg_st = 0

    if restecg == "normal":
        restecg_normal = 1
    elif restecg == "st-t abnormality":
        restecg_st = 1

    input_data = pd.DataFrame({
        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "thalch": [thalch],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "sex_Male": [sex_male],
        "cp_non-anginal": [cp_non_anginal],
        "cp_typical angina": [cp_typical_angina],
        "restecg_normal": [restecg_normal],
        "restecg_st-t abnormality": [restecg_st]
    })

    try:

        prediction = model.predict(input_data)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(input_data).max() * 100

        prediction_map = {
            0: "🟢 No Heart Disease",
            1: "🟡 Mild Heart Disease",
            2: "🟠 Moderate Heart Disease",
            3: "🔴 Severe Heart Disease",
            4: "🚨 Critical Heart Disease"
        }

        st.divider()

        st.subheader("🩺 Prediction Result")

        st.success(prediction_map[prediction])

        if confidence is not None:
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

        st.subheader("💡 Health Recommendations")

        if prediction == 0:
            st.success("""
✅ No significant signs of heart disease detected.

**Recommendations**
- Maintain a healthy balanced diet.
- Exercise for at least 30 minutes daily.
- Avoid smoking and excessive alcohol.
- Get regular health checkups.
""")

        elif prediction == 1:
            st.warning("""
🟡 Mild Heart Disease Risk

**Recommendations**
- Consult a physician.
- Reduce cholesterol and blood pressure.
- Follow a healthy diet.
- Exercise regularly.
""")

        elif prediction == 2:
            st.warning("""
🟠 Moderate Heart Disease Risk

**Recommendations**
- Schedule a cardiology consultation.
- Monitor blood pressure and blood sugar.
- Reduce salt and saturated fats.
- Follow prescribed medications if any.
""")

        elif prediction == 3:
            st.error("""
🔴 Severe Heart Disease Risk

**Recommendations**
- Consult a cardiologist immediately.
- Follow all medical advice carefully.
- Monitor cholesterol and blood pressure.
- Adopt a heart-healthy lifestyle.
""")

        else:
            st.error("""
🚨 Critical Heart Disease Risk

**Recommendations**
- Seek immediate medical attention.
- Visit the nearest hospital or cardiologist.
- Follow emergency medical guidance.
- Do not ignore symptoms such as chest pain or shortness of breath.
""")

        with st.expander("📊 View Encoded Input Data"):
            st.dataframe(input_data, use_container_width=True)

        st.info(
            "⚠️ **Disclaimer:** This prediction is generated using a Machine Learning model "
            "and is intended for educational purposes only. "
            "It should not replace professional medical diagnosis or treatment."
        )

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")