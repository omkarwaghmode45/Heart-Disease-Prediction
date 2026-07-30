import streamlit as st
import pandas as pd
import joblib
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.title("❤️ Heart Disease Prediction System")
st.write(
    "Predict the likelihood of heart disease using patient clinical information."
)
st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
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
        ["normal", "lv hypertrophy", "st-t abnormality"]
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


if st.button("🔍 Predict Heart Disease"):
    sex_male = 1 if sex == "Male" else 0

    fbs = int(fbs)
    exang = int(exang)

    # Chest Pain
    cp_non_anginal = 0
    cp_typical_angina = 0

    if cp == "non-anginal":
        cp_non_anginal = 1
    elif cp == "typical angina":
        cp_typical_angina = 1

    # Rest ECG
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

    prediction = model.predict(input_data)

    prediction_map = {
        0: "🟢 No Heart Disease",
        1: "🟡 Mild Heart Disease",
        2: "🟠 Moderate Heart Disease",
        3: "🔴 Severe Heart Disease",
        4: "🚨 Critical Heart Disease"
    }

    st.success(f"Prediction: {prediction_map[int(prediction[0])]}")