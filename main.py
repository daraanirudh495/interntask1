# train_model.py

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("D:\loan_default_prediction\Loan_default.csv")

# Show basic info
print(df.info())
print(df.head())

# Rename target if needed
if 'Default' in df.columns:
    df.rename(columns={'Default': 'loan_default'}, inplace=True)

# Drop non-numeric or irrelevant columns like IDs
if 'LoanID' in df.columns:
    df.drop('LoanID', axis=1, inplace=True)

# Convert Yes/No categorical fields to binary (0/1)
binary_columns = ['HasCoSigner', 'HasMortgage', 'HasDependents']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Select only numeric fields for the simplified web form
selected_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'HasCoSigner', 'HasMortgage', 'HasDependents'
]

# Check if all required columns are present
missing = [col for col in selected_features + ['loan_default'] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in the dataset: {missing}")

X = df[selected_features]
y = df['loan_default']

# Fill any missing values (if applicable)
X.fillna(X.median(), inplace=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Model and scaler saved successfully!")
