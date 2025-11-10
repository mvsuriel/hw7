"""
Script to train ML models and save them as pickle files
"""
import subprocess
import sys

# Install dependencies
print("Installing dependencies...")
subprocess.check_call([
    "/opt/homebrew/bin/python3.11", "-m", "pip", "install",
    "scikit-learn", "joblib", "pandas", "numpy"
])
print("Dependencies installed successfully!\n")

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Get working directory
wd = os.getcwd()
print(f"Working directory: {wd}")

# Load and prepare data
print("\n" + "=" * 60)
print("Loading and preparing data...")
print("=" * 60)

# Load data
csv_path = os.path.join(wd, "data", "sample_diabetes_mellitus_data.csv")
df = pd.read_csv(csv_path)

# Split data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Encode categorical data first (before splitting)
# For ethnicity
le = LabelEncoder()
df["ethnicity"] = le.fit_transform(df["ethnicity"])

# For gender (binary)
df["gender"] = df["gender"].map({"M": 1, "F": 0})

# Now split the data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Clean data - Remove rows with NaN values in specified columns
cols_nan = ["age", "gender", "ethnicity"]
train_df = train_df.dropna(subset=cols_nan)
test_df = test_df.dropna(subset=cols_nan)

# Fill NaN values with mean in specified columns
cols_fill = ["height", "weight"]
for col in cols_fill:
    mean_value = train_df[col].mean()
    train_df[col] = train_df[col].fillna(mean_value)
    test_df[col] = test_df[col].fillna(mean_value)

# Define features and target
FEATURES = [
    "age", "height", "weight",
    "aids", "cirrhosis", "hepatic_failure",
    "immunosuppression", "leukemia", "lymphoma",
    "solid_tumor_with_metastasis",
]
TARGET = "diabetes_mellitus"

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

# Train models
print("\n" + "=" * 60)
print("Training models...")
print("=" * 60)

# Train Logistic Regression
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)

# Train Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

print(f"Trained: {type(model_lr).__name__}")
print(f"Trained: {type(model_rf).__name__}")

# Save models as pickle files
print("\n" + "=" * 60)
print("Saving trained models as pickle files")
print("=" * 60)

model_lr_path = os.path.join(wd, "model_logistic_regression.pkl")
model_rf_path = os.path.join(wd, "model_random_forest.pkl")

joblib.dump(model_lr, model_lr_path)
joblib.dump(model_rf, model_rf_path)

print(f"\nModels saved successfully:")
print(f"  1. Logistic Regression: {model_lr_path}")
print(f"  2. Random Forest: {model_rf_path}")

# Verify file sizes
lr_size = os.path.getsize(model_lr_path) / 1024  # KB
rf_size = os.path.getsize(model_rf_path) / 1024  # KB

print(f"\nFile sizes:")
print(f"  Logistic Regression: {lr_size:.2f} KB")
print(f"  Random Forest: {rf_size:.2f} KB")

# Test loading the saved model
print(f"\nVerifying saved models...")
loaded_model_lr = joblib.load(model_lr_path)
X_test = test_df[FEATURES]
test_prediction = loaded_model_lr.predict_proba(X_test)[:5, 1]

print(f"  Loaded model type: {type(loaded_model_lr).__name__}")
print(f"  Test prediction sample: {test_prediction}")
print(f"\nModels successfully saved and verified!")
print("=" * 60)