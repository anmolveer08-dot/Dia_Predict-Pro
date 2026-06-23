import numpy as py
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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


#===============Data Visulization ====================

#scatte plot
sns.scatterplot(x='diabetes', y='bmi', data=df)
plt.title("Bmi Acc. To Diabetes")
plt.show()



#===============boxplot plot ======================
sns.boxplot(x='diabetes', y='age', data=df)
plt.title("Diabetes on the basis of Age ")
plt.show()


#=========Histogram plot===========

plt.hist(df['age'], bins=20)
plt.title('age')
plt.xlabel('age')
plt.ylabel('Frequency')
plt.show()

#=================Heatmap plot===========================

sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')

plt.show()

#liner regression 
le=LabelEncoder()

df['gender']= le.fit_transform(df['gender'])
df['smoking_history']=le.fit_transform(df['smoking_history'])


X=df.drop('diabetes', axis=1)
y=df['diabetes']


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 , random_state= 42)

model = LinearRegression()

model.fit(X_train,y_train)

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)

# Evaluate Model

print("R² Score:", r2_score(y_test, y_pred))

print(y_pred[:10])


#============ logstic regression =====================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#encding 

le=LabelEncoder()

df['gender']= le.fit_transform(df['gender'])
df['smoking_history']=le.fit_transform(df['smoking_history'])


X=df.drop('diabetes', axis=1)
y=df['diabetes']


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 , random_state= 42)

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)


#========train and test accuracy======

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#===== Evaluate Model
print("Train:", model.score(X_train, y_train))
print("Test :", model.score(X_test, y_test))




# ========classfication report=====Confusion Matrix & Visulazition =======


print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print(cm)


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



#=========== decision tree =============

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Encoding
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

x=df.drop('diabetes', axis=1)
y = df['diabetes']

#Traning and testing of model
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 ,random_state=42)


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


#============== Decision Tree regression =======

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree

x=df.drop('diabetes', axis=1)
y = df['diabetes']

#Traning and testing of model
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 ,random_state=42)


model=DecisionTreeRegressor( random_state=42, max_depth= 5)


model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# ========Evaluate Model

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# tree ploting

plt.figure(figsize=(15,8))
plot_tree(model, filled=True)
plt.show()



#======================= Random Forest Regression =====================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



x=df.drop('diabetes', axis=1)
y = df['diabetes']

#Traning and testing of model
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2 ,random_state=42)

model=RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# ======Evaluate Model

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred)**0.5)
print("R2 Score:", r2_score(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False)) 


import matplotlib.pyplot as plt

importance = importance.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()



# #================== Random Forest classifier =========

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===Evaluate Model

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



#=================== KNNN regression ================

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=KNeighborsRegressor(n_neighbors=5)


model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# Evaluate Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred)**0.5)
print("R2 Score:", r2_score(y_test, y_pred))

# best value for k
for k in range(1,21):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    print(k, model.score(X_test, y_test))


    #=================== KNN classification =================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Evaluate Model

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========KNN Classification Confusion Matrix ===========
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()

import pickle

# Save Model
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Label Encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)

print("Model and Encoder saved successfully!")