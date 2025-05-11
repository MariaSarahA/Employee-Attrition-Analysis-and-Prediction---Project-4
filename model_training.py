# model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load your dataset
df = pd.read_csv('data/employee_data.csv')

# Drop unnecessary columns if any (adjust as needed)
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('Attrition', axis=1))
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Save the column names used in training
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(df_encoded.columns.tolist(), 'models/features.pkl')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
joblib.dump(scaler, 'models/scaler.pkl')

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
joblib.dump(model, 'models/random_forest_model.pkl')

print("✅ Model, scaler, and features saved successfully!")


