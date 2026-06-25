import streamlit as st
import numpy as np
import joblib
import base64

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("random_forest.pkl")

# ==========================
# BACKGROUND IMAGE
# ==========================
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("C:/Users/asus/OneDrive/Documents/GitHub/Dia_Predict-Pro/bg.jpg")

st.markdown(f"""
<style>

.stApp {{
    background-image: url("data:image/jpg;base64,{bg}");
    background-size: cover;
    background-position: center;
}}

.main {{
    background: rgba(0,0,0,0.45);
    padding: 20px;
    border-radius: 20px;
}}

.title {{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}}

.acc {{
    text-align:center;
    font-size:18px;
    color:#00ff88;
}}

</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================
st.markdown(
    '<p class="title">🩺 Smart Diabetes Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="acc">Random Forest Accuracy : 91.2%</p>',
    unsafe_allow_html=True
)

st.divider()

# ==========================
# INPUTS
# ==========================
col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 1, 120, 25)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        ["never", "former", "current"]
    )

    bmi = st.number_input("BMI", value=25.0)

    hba1c = st.number_input("HbA1c", value=5.5)

    insulin_level = st.number_input(
        "Insulin Level",
        value=15.0
    )

    heart_rate = st.number_input(
        "Heart Rate",
        value=72
    )

    cholesterol_total = st.number_input(
        "Cholesterol Total",
        value=180.0
    )

    diabetes_risk_score = st.number_input(
        "Diabetes Risk Score",
        value=50.0
    )

with col2:

    alcohol_consumption_per_week = st.number_input(
        "Alcohol Consumption Per Week",
        value=0.0
    )

    physical_activity_minutes_per_week = st.number_input(
        "Physical Activity Minutes Per Week",
        value=150
    )

    diet_score = st.number_input(
        "Diet Score",
        value=5.0
    )

    sleep_hours_per_day = st.number_input(
        "Sleep Hours Per Day",
        value=8.0
    )

    family_history_diabetes = st.selectbox(
        "Family History Diabetes",
        [0, 1]
    )

    hypertension_history = st.selectbox(
        "Hypertension History",
        [0, 1]
    )

    cardiovascular_history = st.selectbox(
        "Cardiovascular History",
        [0, 1]
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        value=80.0
    )

    glucose_fasting = st.number_input(
        "Glucose Fasting",
        value=90.0
    )

# ==========================
# ENCODING
# ==========================
gender = 1 if gender == "Male" else 0

smoking_dict = {
    "never": 0,
    "former": 1,
    "current": 2
}

smoking_status = smoking_dict[smoking_status]

# ==========================
# PREDICTION
# ==========================
if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
):

    features = np.array([[
        age,
        gender,
        smoking_status,
        alcohol_consumption_per_week,
        physical_activity_minutes_per_week,
        diet_score,
        sleep_hours_per_day,
        family_history_diabetes,
        hypertension_history,
        cardiovascular_history,
        bmi,
        diastolic_bp,
        heart_rate,
        cholesterol_total,
        glucose_fasting,
        insulin_level,
        hba1c,
        diabetes_risk_score
    ]])

    prediction = model.predict(features)

    st.write("")

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")