# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load saved model, scaler, and feature names
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/features.pkl')

st.title("🔍 Employee Attrition Prediction")
st.markdown("Predict the likelihood of an employee leaving the company.")

# Sidebar inputs
st.sidebar.header("Employee Info")
user_input = {
    'Age': st.sidebar.slider('Age', 18, 60, 30),
    'BusinessTravel': st.sidebar.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']),
    'DailyRate': st.sidebar.slider('Daily Rate', 100, 1500, 500),
    'Department': st.sidebar.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources']),
    'DistanceFromHome': st.sidebar.slider('Distance From Home', 1, 30, 10),
    'Education': st.sidebar.slider('Education Level', 1, 5, 3),
    'EducationField': st.sidebar.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']),
    'EnvironmentSatisfaction': st.sidebar.slider('Environment Satisfaction', 1, 4, 3),
    'Gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
    'HourlyRate': st.sidebar.slider('Hourly Rate', 30, 100, 60),
    'JobInvolvement': st.sidebar.slider('Job Involvement', 1, 4, 3),
    'JobLevel': st.sidebar.slider('Job Level', 1, 5, 2),
    'JobRole': st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                                  'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                                  'Sales Representative', 'Research Director', 'Human Resources']),
    'JobSatisfaction': st.sidebar.slider('Job Satisfaction', 1, 4, 3),
    'MaritalStatus': st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced']),
    'MonthlyIncome': st.sidebar.slider('Monthly Income', 1000, 20000, 5000),
    'MonthlyRate': st.sidebar.slider('Monthly Rate', 2000, 25000, 10000),
    'NumCompaniesWorked': st.sidebar.slider('Num Companies Worked', 0, 10, 2),
    'OverTime': st.sidebar.selectbox('Over Time', ['Yes', 'No']),
    'PercentSalaryHike': st.sidebar.slider('Percent Salary Hike', 10, 25, 15),
    'PerformanceRating': st.sidebar.slider('Performance Rating', 1, 4, 3),
    'RelationshipSatisfaction': st.sidebar.slider('Relationship Satisfaction', 1, 4, 3),
    'StockOptionLevel': st.sidebar.slider('Stock Option Level', 0, 3, 1),
    'TotalWorkingYears': st.sidebar.slider('Total Working Years', 0, 40, 10),
    'TrainingTimesLastYear': st.sidebar.slider('Training Times Last Year', 0, 6, 3),
    'WorkLifeBalance': st.sidebar.slider('Work Life Balance', 1, 4, 3),
    'YearsAtCompany': st.sidebar.slider('Years At Company', 0, 40, 5),
    'YearsInCurrentRole': st.sidebar.slider('Years In Current Role', 0, 18, 4),
    'YearsSinceLastPromotion': st.sidebar.slider('Years Since Last Promotion', 0, 15, 2),
    'YearsWithCurrManager': st.sidebar.slider('Years With Current Manager', 0, 17, 3)
}

# Convert to dataframe and encode
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# Scale
scaled_input = scaler.transform(input_encoded)

# Predict
prediction = model.predict(scaled_input)[0]
pred_prob = model.predict_proba(scaled_input)[0][1]

st.subheader("Prediction")
if prediction == 1:
    st.error(f"⚠️ The employee is likely to leave. (Probability: {pred_prob:.2%})")
else:
    st.success(f"✅ The employee is likely to stay. (Probability: {1 - pred_prob:.2%})")

st.caption("Note: This is a predictive tool based on historical data. Use results cautiously.")
