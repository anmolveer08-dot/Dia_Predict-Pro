 #================== Random Forest classifier =========

import numpy as py
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


df=pd.read_csv('diabetes_prediction_dataset.csv')
print(df)

# Missing values
print(df.isnull().sum())

# Number of unique values
print(df.nunique())

print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))

# Correlation (numeric columns only)
print(df.corr(numeric_only=True))

print(df.dtypes)


X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
print("Accuracy:", accuracy_score(y_test, y_pred))



# Confusion Matrix Of Random Forest Classification 

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Important Feature

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print(importance)