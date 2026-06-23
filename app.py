import streamlit as st
import pandas as pd
import pickle

# Load model and feature list
model = pickle.load(open("diabetes_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["No Info", "current", "ever", "former", "never", "not current"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    HbA1c_level = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=300,
        value=100
    )

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    data = {feature: 0 for feature in features}

    data["age"] = age
    data["hypertension"] = hypertension
    data["heart_disease"] = heart_disease
    data["bmi"] = bmi
    data["HbA1c_level"] = HbA1c_level
    data["blood_glucose_level"] = blood_glucose_level

    # Gender Encoding
    if gender == "Male":
        data["gender_Male"] = 1
    elif gender == "Other":
        data["gender_Other"] = 1

    # Smoking Encoding
    if smoking_history == "current":
        data["smoking_history_current"] = 1
    elif smoking_history == "ever":
        data["smoking_history_ever"] = 1
    elif smoking_history == "former":
        data["smoking_history_former"] = 1
    elif smoking_history == "never":
        data["smoking_history_never"] = 1
    elif smoking_history == "not current":
        data["smoking_history_not current"] = 1

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🔴 High Risk of Diabetes")
    else:
        st.success("🟢 Low Risk of Diabetes")