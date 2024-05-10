import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the first 5 rows of the data
data.head()
print(data.head())

# Seperating Categorical and Numerical Columns
cat_col = data.select_dtypes(include='object').columns
num_col = data.select_dtypes(exclude='object').columns

# Display the Categorical Columns
print(cat_col)

# Display the Numerical Columns
print(num_col)

# checking for missing values
print(data.isnull().sum())

# Display the unique values of the Categorical Columns
for col in cat_col:
    print(col)
    print(data[col].unique())

# Display the unique values of the Numerical Columns
for col in num_col:
    print(col)
    print(data[col].unique())

# converting categorical columns to numerical columns using Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_col:
    data[col] = le.fit_transform(data[col])

# Display the first 5 rows of the data
data.head()
print(data.head())

# concatenating the Categorical and Numerical Columns
new_data = pd.concat([data[cat_col], data[num_col]], axis=1)
print(new_data.head())
print(new_data.isna().sum())

# Splitting the data into Features and Target
X = new_data.drop('Attrition', axis=1)
y = new_data['Attrition']

# Splitting the data into Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the Target
y_pred = model.predict(X_test)

# Evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

