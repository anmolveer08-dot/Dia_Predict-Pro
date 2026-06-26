#== random forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib #py -m streamlit run Dia_Pro.py

from sklearn.preprocessing  import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df=pd.read_csv('Diabetes_and_LifeStyle_Dataset -selected-columns.csv')

print(df)
print(df['diagnosed_diabetes'].value_counts())
print(df.shape)


print(df.head())
print(df.info())

print(df['diagnosed_diabetes'].value_counts())
print(df.describe())


# Encoding

gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

df['gender'] = gender_encoder.fit_transform(df['gender'])
df['smoking_status'] = smoking_encoder.fit_transform(df['smoking_status'])

joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(smoking_encoder, "smoking_encoder.pkl")


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
    X, y, test_size=0.2, random_state=42,stratify=y
)

model=RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train,y_train)

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

joblib.dump(model, 'random_forest.pkl')

y_pred=model.predict(X_test)


#Elavute the model

print("Acccuracy Train",model.score(X_train,y_train))
print("Accuracy Test",model.score(X_test,y_test))




#confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Importace feature
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance)

sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('Feature Importance')
plt.show()