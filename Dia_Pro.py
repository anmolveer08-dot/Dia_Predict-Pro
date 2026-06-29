import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

model = joblib.load("random_forest.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
smoking_encoder = joblib.load("smoking_encoder.pkl")
importance = joblib.load("feature_importance.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Dia Predict Pro",
    page_icon="🩺",
    layout="wide"
)

col1, col2 = st.columns([8, 1])

with col1:
    st.title("🩺 Dia Predict Pro")
    st.write("AI-powered Diabetes Prediction System")

with col2:
    st.image("logo.jpg", width=100)

# ==========================
# BACKGROUND IMAGE
# ==========================

import base64
import streamlit as st

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("imgback.jpg")

st.markdown(f"""
<style>

.stApp {{
    background: transparent;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;

    background: url("data:image/jpeg;base64,{img}") center center;
    background-size: cover;
    background-repeat: no-repeat;

    filter: blur(5px) brightness(55%) contrast(90%);
    transform: scale(1.08);

    z-index: -1;
}}

</style>
""", unsafe_allow_html=True)

# styling 
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
# TITLE   '<p class="acc">Random Forest Accuracy : 92.96%%</p>',
# ==========================

st.markdown(
    f'<div class="accuracy">{'Random Forest Classification '} Accuracy : {92.96:.2f}%</div>',
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


# =====================================================
# MODEL COMPARISON
# =====================================================

    st.markdown(
        "<h1 class='main-title'>📊 Model Comparison</h1>",
        unsafe_allow_html=True
    )

    st.write("Performance comparison of different Machine Learning models.")

    st.divider()

    # ==============================
    # Metrics
    # ==============================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🏆 Best Model", "Random Forest")
    c2.metric("🎯 Accuracy", "92.96%")
    c3.metric("📈 Precision", "1.00")
    c4.metric("📉 Recall", "0.87")

    st.divider()

    # ==============================
    # Model Table
    # ==============================

    comparison = pd.DataFrame({

        "Model":[
            "Random Forest",
            "Logistic Regression",
            "KNN",
            "Decision Tree"
        ],

        "Accuracy":[
            92.96,
            86.45,
            85.36,
            70.70
        ],

        "Precision":[
            1.00,
            0.88,
            0.89,
            0.81
        ],

        "Recall":[
            0.87,
            0.89,
            0.87,
            0.83
        ],

        "F1 Score":[
            0.93,
            0.89,
            0.88,
            0.82
        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    st.subheader("📈 Accuracy Comparison")

    chart = comparison.set_index("Model")["Accuracy"]

    st.bar_chart(chart)

    st.success(
        "🏆 Random Forest achieved the highest accuracy and was selected as the final deployed model."
    )

    # =====================================================
# FEATURE IMPORTANCE
# =====================================================

    st.markdown(
        "<h1 class='main-title'>📈 Feature Importance</h1>",
        unsafe_allow_html=True
    )

    st.write(
        "Feature importance calculated using the Random Forest model."
    )

    st.divider()

    feature_names = [

        "Age",
        "Gender",
        "Smoking",
        "BMI",
        "Physical Activity",
        "Family History",
        "Cholesterol",
        "Glucose",
        "HbA1c",
        "Risk Score",
        "Insulin",
        "Diet Score"

    ]

    importance_df = pd.DataFrame({

        "Feature":feature_names,

        "Importance":importance

    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("📊 Feature Importance Chart")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.divider()

    st.subheader("🔝 Top 5 Important Features")

    top5 = importance_df.head(5)

    for _, row in top5.iterrows():

        st.write(
            f"✅ **{row['Feature']}** : {row['Importance']:.3f}"
        )

    st.info(
        """
        Higher importance means that the feature contributes
        more to the prediction made by the Random Forest model.
        """
    )