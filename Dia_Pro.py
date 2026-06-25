import streamlit as st
import joblib
import numpy as np
import base64
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DIA-PREDICT-PRO",
    page_icon="🩺",
    layout="wide"
)

# ---------------- BACKGROUND ----------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        .main {{
            background: rgba(255,255,255,0.85);
            border-radius: 15px;
            padding: 20px;
        }}

        h1 {{
            text-align:center;
            color:#003366;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("bg.jpg")

# ---------------- LOAD MODELS ----------------
lr_model = joblib.load("logistic.pkl")
dt_model = joblib.load("decision_tree.pkl")
rf_model = joblib.load("random_forest.pkl")
knn_model = joblib.load("knn.pkl")

ann_model = load_model("ann_model.h5")

scaler = joblib.load("scaler.pkl")

# ---------------- ACCURACY VALUES ----------------
accuracies = {
    "Logistic Regression": 86.5,
    "Decision Tree": 84.2,
    "Random Forest": 89.7,
    "KNN": 87.1,
    "ANN": 91.2
}

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1>🩺 DIA-PREDICT-PRO</h1>
    <h4 style='text-align:center;color:#444'>
    Smart Diabetes Prediction System
    </h4>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("Patient Information")

    age = st.number_input("Age", 1, 120)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=1.0,
        max_value=15.0
    )

with col2:

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever"
        ]
    )

# ---------------- MODEL SELECTION ----------------
st.subheader("Choose Prediction Model")

selected_model = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "ANN"
    ]
)

# ---------------- ENCODING ----------------
gender = 1 if gender == "Male" else 0

smoking_dict = {
    "never":0,
    "former":1,
    "current":2,
    "not current":3,
    "ever":4
}

smoking_status = smoking_dict[smoking_status]

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Diabetes Risk"):

    data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_status,
        bmi,
        hba1c,
        blood_glucose
    ]])

    data = scaler.transform(data)

    if selected_model == "Logistic Regression":
        prediction = lr_model.predict(data)

    elif selected_model == "Decision Tree":
        prediction = dt_model.predict(data)

    elif selected_model == "Random Forest":
        prediction = rf_model.predict(data)

    elif selected_model == "KNN":
        prediction = knn_model.predict(data)

    else:
        prediction = ann_model.predict(data)
        prediction = (prediction > 0.5).astype(int)

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 High Risk of Diabetes")
    else:
        st.success("🟢 Low Risk of Diabetes")

    st.info(f"Model Used: {selected_model}")

    st.info(
        f"Model Accuracy: {accuracies[selected_model]}%"
    )