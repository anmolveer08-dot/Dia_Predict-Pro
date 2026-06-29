#======== KNN Classification ==============

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df=pd.read_csv('Diabetes_and_LifeStyle_Dataset -selected-columns.csv')

print(df)

print(df.head())
print(df.shape)

print(df.info())
print(df.describe())


# Encoding
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Features and Target
X = df[[
    'Age',
    'gender',
    'smoking_status',
    'bmi',
    'physical_activity_minutes_per_week',
    'family_history_diabetes',
    'cholesterol_total',
    'glucose_fasting',
    'hba1c',
    'diabetes_risk_score',
    'insulin_level',
    'diet_score'
]]

y = df['diagnosed_diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = KNeighborsClassifier(
    n_neighbors=18,
    weights='distance',
    metric='manhattan'
)

model.fit(X_train, y_train)

joblib.dump(model, 'knn.pkl')

y_pred = model.predict(X_test)

print(y.value_counts(normalize=True))

print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(scaler, 'scaler.pkl')