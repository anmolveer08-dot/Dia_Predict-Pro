# 🩺 Smart Diabetes Prediction System

## 📌 Project Overview

The Smart Diabetes Prediction System is a Machine Learning-based web application developed using Streamlit. The system predicts whether a person is at risk of diabetes based on medical and lifestyle information provided by the user.

The application allows users to select different machine learning models and instantly receive prediction results.

---

## 🚀 Features

* Interactive and user-friendly Streamlit interface
* Diabetes risk prediction
* Multiple Machine Learning models

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * K-Nearest Neighbors (KNN)
  * Artificial Neural Network (ANN)
* Model accuracy comparison
* Welcome page with modern UI
* Background image support
* Real-time predictions

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Streamlit
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Joblib
* Matplotlib
* Seaborn

---

## 📂 Project Structure

```text
Dia_Predict-Pro/
│
├── Dia_Pro.py
├── random_forest.pkl
├── decision_tree.pkl
├── logistic.pkl
├── knn.pkl
├── scaler.pkl
├── bg.png
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

The prediction model uses the following features:

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

---

## ⚙️ Installation

### Clone Repository

```bash
 clone https://github.com/yourusername/Dia_Predict-Pro.git
```

### Navigate to Project Folder

```bash
cd Dia_Predict-Pro
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python -m streamlit run Dia_Pro.py
```

After running, open:

```text
http://localhost:8501
```

---

## 📈 Machine Learning Models

The project evaluates multiple machine learning algorithms:

| Model                 | Purpose                       |
| -------------------   | ----------------------------- |
| Logistic Regression   | Linear Classification         |
| Decision Tree (best)  | Tree-Based Classification     |
| Random Forest         | Ensemble Learning             |
| KNN                   | Distance-Based Classification |
| ANN                   | Deep Learning Classification  |

---

## 🎯 Objective

The objective of this project is to assist in the early detection of diabetes by leveraging machine learning techniques and providing a simple interface for healthcare prediction.

## 🔮 Future Enhancements

* Patient report generation
* PDF export
* Cloud deployment
* Explainable AI (XAI)
* Real-time healthcare integration
* Additional disease prediction modules

---

## 👨‍💻 Author

Developed as a Machine Learning Project using Python and Streamlit.

---

## 📄 License

This project is developed for educational and research purposes.
