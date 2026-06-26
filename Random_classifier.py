#== random forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Features and Target
X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model=RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train,y_train)

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