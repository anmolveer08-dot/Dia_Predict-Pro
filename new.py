#  #================== Random Forest classifier =========

# import numpy as py
# import pandas as pd
# import seaborn as sns 
# import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report,confusion_matrix


# df=pd.read_csv('diabetes_prediction_dataset.csv')
# print(df)

# # Missing values
# print(df.isnull().sum())

# # Number of unique values
# print(df.nunique())

# print(df.duplicated().sum())
# print(df.drop_duplicates(inplace=True))

# # Correlation (numeric columns only)
# print(df.corr(numeric_only=True))

# print(df.dtypes)


# X = df.drop('heart_disease', axis=1)
# y = df['heart_disease']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = RandomForestClassifier(n_estimators=100, random_state=42)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Train Accuracy:", model.score(X_train, y_train))
# print("Test Accuracy:", model.score(X_test, y_test))
# print("Accuracy:", accuracy_score(y_test, y_pred))



# # Confusion Matrix Of Random Forest Classification 

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Random Forest Confusion Matrix")
# plt.show()

# # Important Feature

# importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': model.feature_importances_
# })

# importance = importance.sort_values(by='Importance', ascending=False)

# print(importance)

import numpy as py
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df=pd.read_csv('diabetes_prediction_dataset.csv')# Data loading 
print(df)


df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encoding
df = pd.get_dummies(
    df,
    columns=['gender', 'smoking_history'],
    drop_first=True,
    dtype=int
)
print(df.head())

# Features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

print(X.columns.tolist())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(     #== without tuning 97% == without 94%==in classifeir mode 
    
    max_depth=7,
    random_state=42
)
model.fit(X_train,y_train)

# Model Prediction 
y_pred=model.predict(X_test)

#========================= Evaluate Model

print("Train Accuracy:" ,model.score(X_train,y_train))
print("Test Accuracy:", model.score(X_test,y_test))
print(classification_report(y_test, y_pred))


# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()


#==== Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print(importance.sort_values(by='Importance', ascending=False))




plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True
)
plt.show()