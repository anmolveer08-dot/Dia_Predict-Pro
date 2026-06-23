# Dia_Predict-Pro
DiaPredict Pro is a machine learning-powered web application that predicts diabetes risk using patient health data and provides real-time predictions through an intuitive Streamlit interface


# 🩺 DiaPredict Pro

### Advanced Diabetes Risk Prediction System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

## 📌 Project Overview

DiaPredict Pro is a machine learning-powered healthcare application designed to predict the likelihood of diabetes using patient health and lifestyle information. The system analyzes multiple medical factors and provides instant predictions through an interactive Streamlit dashboard.

The project demonstrates the practical application of Data Science, Machine Learning, and Healthcare Analytics to support early diabetes risk assessment and preventive healthcare decision-making.

---

## 🎯 Objectives

* Analyze diabetes-related health data.
* Perform Exploratory Data Analysis (EDA).
* Train and compare multiple Machine Learning models.
* Select the best-performing model based on evaluation metrics.
* Deploy the final model through a professional Streamlit web application.

---

## 📊 Dataset Features

The model uses the following patient attributes:

| Feature             | Description               |
| ------------------- | ------------------------- |
| Gender              | Male/Female               |
| Age                 | Patient Age               |
| Hypertension        | Presence of Hypertension  |
| Heart Disease       | Presence of Heart Disease |
| Smoking History     | Smoking Status            |
| BMI                 | Body Mass Index           |
| HbA1c Level         | Glycated Hemoglobin Level |
| Blood Glucose Level | Blood Sugar Measurement   |

### Target Variable

**Diabetes**

* 0 → No Diabetes
* 1 → Diabetes

---

## 🔍 Exploratory Data Analysis

The project includes:

* Scatter Plot Analysis
* Box Plot Analysis
* Histogram Visualization
* Correlation Heatmap
* Feature Importance Analysis
* Confusion Matrix Visualization
* Decision Tree Visualization

---

## 🤖 Machine Learning Models Evaluated

The following algorithms were trained and evaluated:

* Linear Regression
* Logistic Regression
* Decision Tree Classifier
* Decision Tree Regressor
* Random Forest Classifier
* Random Forest Regressor
* K-Nearest Neighbors (KNN) Classifier
* K-Nearest Neighbors (KNN) Regressor

---

## 🏆 Best Performing Model

After evaluating all models, the **Decision Tree Classifier** achieved the highest predictive performance and was selected for deployment.

### Final Model Configuration

```python
DecisionTreeClassifier(
    max_depth=7,
    random_state=42
)
```

### Why Decision Tree?

✅ Highest test accuracy among evaluated models

✅ Easy to interpret and visualize

✅ Handles non-linear relationships effectively

✅ Provides feature importance insights

✅ Fast prediction speed for deployment

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Pickle

---

## 🖥️ Application Workflow

1. User enters health information.
2. Input data is preprocessed.
3. Decision Tree model analyzes the input.
4. Diabetes risk prediction is generated.
5. Results are displayed instantly through the dashboard.

---

## 📂 Project Structure

```text
DiaPredict-Pro/
│
├── app.py
├── diabetes_model.pkl
├── diabetes_prediction_dataset.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🚀 Installation & Usage

### Clone Repository

```bash
git clone https://github.com/yourusername/DiaPredict-Pro.git
```

### Navigate to Project Folder

```bash
cd DiaPredict-Pro
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📸 Dashboard Preview

Add screenshots of your Streamlit application here.

Example:

* Home Dashboard
* Input Form
* Prediction Results
* Risk Assessment Screen

---

## 🔮 Future Improvements

* Hyperparameter Optimization
* Probability-Based Predictions
* Cloud Deployment
* Medical Report Generation
* Advanced Data Analytics
* User Authentication System

---

## 👨‍💻 Author

**Singh**

Machine Learning & Data Science Enthusiast

---

## 📄 License

This project is intended for educational and learning purposes. Feel free to use and modify it for academic and personal projects.

---

### ⭐ If you found this project helpful, please consider giving it a star on GitHub!

**DiaPredict Pro — Predicting Diabetes Risk with the Power of Machine Learning.**

