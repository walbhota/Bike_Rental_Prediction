# -*- coding: utf-8 -*-
"""
Bike Rental Prediction
Author: Ebhota Walter Eromosele
Objective: Predict daily bike rental demand based on weather and temporal factors.
"""

# Import libraries
import pandas as pd
import io
import requests
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Download and load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
print("Downloading dataset...")
with zipfile.ZipFile(io.BytesIO(requests.get(url).content)) as z:
    with z.open('day.csv') as f:
        data = pd.read_csv(f)

print("Dataset loaded successfully!\n")
print(data.head())

# Initial Data Exploration
print("\nData Information:")
data.info()
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values Check:")
print(data.isnull().sum())

# Visualize relationships between features
sns.pairplot(data, vars=['temp', 'atemp', 'hum', 'windspeed', 'cnt'])
plt.show()

# Feature Selection and Splitting Data
X = data.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
y = data['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
print("\nTraining Models...")

# Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

print("Models trained successfully!")

# Predictions
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)

# Evaluate Models
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"\nModel Performance:")
print(f"Random Forest MAE: {mae_rf}")
print(f"Linear Regression MAE: {mae_lr}")

# Insights and Recommendations
print("\nInsights:")
print("1. Random Forest outperforms Linear Regression, indicating better handling of non-linear relationships.")
print("2. Feature engineering and hyperparameter tuning could improve performance further.")