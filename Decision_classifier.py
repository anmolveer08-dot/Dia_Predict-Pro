#===== 
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('Diabetes_and_LifeStyle_Dataset -selected-columns.csv')

print(df)
print(df.shape)

print(df.head())
print(df.info())
print(df.describe())


#Encoding 

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])


X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


model = DecisionTreeClassifier(
    max_depth=5,
     class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, 'decision_tree.pkl')

y_pred = model.predict(X_test)

# ================== model accuracy and output=============

print("Accuracy: train",model.score(X_train,y_train))
print("Accuracy: train",model.score(X_test,y_test))

print(classification_report(y_test, y_pred))

#===========confusion matrix===========

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#==========Tree ploting===========

plt.figure(figsize=(15,8))
plot_tree(model, filled=True)
plt.show()

