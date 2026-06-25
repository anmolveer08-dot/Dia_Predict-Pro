import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load Dataset
df = pd.read_csv('Diabetes_and_LifeStyle_Dataset -selected-columns.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# data visulization

sns.countplot(x='diagnosed_diabetes', data=df)
plt.title('Diabetes Distribution')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.boxplot(x='diagnosed_diabetes', y='Age', data=df)
plt.title('Age vs Diabetes')
plt.show()

sns.boxplot(x='diagnosed_diabetes', y='hba1c', data=df)
plt.title('HbA1c Level vs Diabetes')
plt.show()

df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# Encoding
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Features and Target
X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Training
model.fit(X_train, y_train)

joblib.dump(model, 'logistic.pkl')

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class Distribution
print("\nClass Distribution:")
print(y.value_counts())

