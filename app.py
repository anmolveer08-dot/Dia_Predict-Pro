import pickle
model = pickle.load(open("diabetes_model.pkl", "rb"))

gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))

smoking_encoder = pickle.load(open("smoking_encoder.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))  # only if KNN
# Load Model

# Load Encoder
with open("label_encoder.pkl", "rb") as file:
    le = pickle.load(file)

import streamlit as st

st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------- CUSTOM CSS ----------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    text-align:center;
    color:#0E4D92;
}

.block-container {
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------

st.title("🏥 AI Diabetes Risk Predictor")

st.markdown(
    "Predict the likelihood of diabetes using medical and lifestyle indicators."
)

st.divider()

# ---------- PERSONAL INFO ----------

st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

# ---------- MEDICAL INFO ----------

st.subheader("🩺 Medical Information")

col3, col4, col5 = st.columns(3)

with col3:
    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

with col4:
    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

with col5:
    smoking_history = st.selectbox(
        "Smoking History",
        ["never","former","current","not current"]
    )

# ---------- HEALTH DATA ----------

st.subheader("📊 Health Measurements")

col6, col7, col8 = st.columns(3)
with col6:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

if bmi < 18.5:
    category = "Underweight"

elif bmi < 25:
    category = "Normal Weight"

elif bmi < 30:
    category = "Overweight"

else:
    category = "Obese"

st.metric(
    label="BMI Category",
    value=category
)
    

with col7:
    hba1c = st.number_input(
        "HbA1c Level",
        3.0,
        15.0,
        5.5
    )

with col8:
    blood_glucose = st.number_input(
        "Blood Glucose Level",
        50,
        400,
        120
    )

st.divider()

# ---------- PREDICT BUTTON ----------

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    gender_encoded = gender_encoder.transform([gender])[0]
    smoking_encoded = smoking_encoder.transform([smoking_history])[0]

    data = [[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        smoking_encoded,
        bmi,
        hba1c,
        blood_glucose
    ]]

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("🔴 High Risk of Diabetes")
    else:
        st.success("🟢 Low Risk of Diabetes")