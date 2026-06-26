import streamlit as st
import numpy as np
import joblib
import base64

model = joblib.load("random_forest.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
smoking_encoder = joblib.load("smoking_encoder.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Dia Predict Pro",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# BACKGROUND IMAGE
# ==========================

import base64
import streamlit as st

def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("imgback.jpg")   # change to your file name
st.markdown("""
<style>

/* Main Title */
.main-title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#003366;
    text-shadow:2px 2px 4px white;
    margin-bottom:20px;
}

/* Section Titles */
.section-title{
    text-align:left;
    font-size:30px;
    font-weight:bold;
    color:#FFFFFF;
    text-shadow:2px 2px 4px black;
    margin-top:20px;
}

/* Labels */
label{
    color:##00FF7F !important;
    font-size:18px !important;
    font-weight:bold !important;
}

/* Accuracy Text */
.accuracy{
    text-align:center;
    color:#00FF7F;
    font-size:22px;
    font-weight:bold;
}

/* Result Text */
.result{
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE   '<p class="acc">Random Forest Accuracy : 91.2%</p>',
# ==========================
st.markdown(
    '<div class="main-title">🩺 Dia Predict Pro </div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="accuracy">{'Random Forest Accuracy'} Accuracy : {91.2:.2f}%</div>',
    unsafe_allow_html=True
)

st.divider()
# ==========================
# INPUTS
# ==========================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["Never", "Former", "Current"]
)

bmi = st.number_input(
    "BMI",
    value=25.0
)

physical_activity_minutes_per_week = st.number_input(
    "Physical Activity Minutes Per Week",
    value=150
)

family_history_diabetes = st.selectbox(
    "Family History of Diabetes",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

cholesterol_total = st.number_input(
    "Total Cholesterol",
    value=180.0
)

glucose_fasting = st.number_input(
    "Fasting Glucose",
    value=90.0
)

hba1c = st.number_input(
    "HbA1c Level",
    value=5.5
)

insulin_level = st.number_input(
    "Insulin Level",
    value=15.0
)

diabetes_risk_score = st.number_input(
    "Diabetes Risk Score",
    value=50.0
)

diet_score = st.number_input(
    "Diet Score",
    value=5.0
)



# ==========================
# ENCODING
# ==========================

gender = gender_encoder.transform([gender])[0]
smoking_status = smoking_encoder.transform([smoking_status])[0]
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
    bmi,
    physical_activity_minutes_per_week,
    family_history_diabetes,
    cholesterol_total,
    glucose_fasting,
    hba1c,
    diabetes_risk_score,
    insulin_level,
    diet_score
]])

    prediction = model.predict(features)

    # Debug output
    st.write("Prediction:", prediction)
    st.write("Features:", features)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")